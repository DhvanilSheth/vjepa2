# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""V-JEPA 3: optional changes to what a clip's masks hide, and to how much
each hidden token counts.

OFF BY DEFAULT.  With no key set, `build_mask_collator` returns V-JEPA 2.1's
own collator (src/masks/multiseq_multiblock3d.py), untouched.

All keys go on an entry of the config's `mask:` list.

truncation: tail | random
    Every clip in a batch must end up with the same number of visible tokens
    and the same number of target tokens, so each list is cut to the shortest
    one in the batch.  The lists are sorted by time, and V-JEPA 2.1 cuts from
    the END ("tail").  The effect, measured on the proxy recipe: about a
    quarter of all tokens are neither shown nor asked about, nearly all of
    them in the last two of eight time-steps.  "random" keeps a random subset
    of the same size instead, so the whole clip is used evenly.

hide_time_steps: 0 | k
    0 keeps V-JEPA's spatial blocks, which hide the same region in every
    frame.  k > 0 replaces them for this mask: k whole time-steps, chosen at
    random per clip, are hidden completely and become the prediction targets;
    every token of the other time-steps is visible.  The model then has to
    produce a moment it was not shown from the moments around it.  One
    time-step is `tubelet_size` frames.  Counts are equal for every clip, so
    nothing is cut.

target_weight: none | unexplained      (with target_weight_full, default 0.064)
    Needs hide_time_steps > 0.  "unexplained" gives every hidden token of this
    mask a weight in [0, 1]: how much the shown time-steps fail to explain it
    once the motion most of the frame shares is undone.  The weights are
    computed here, in the data-loading workers, and travel with the batch as a
    fourth item; train.py uses them in the loss.  See app/vjepa_3/change.py.
"""

import numpy as np
import torch

from app.vjepa_3.change import resolve_target_weights, target_weights
from src.masks.multiseq_multiblock3d import MaskCollator, _MaskGenerator

TRUNCATIONS = ("tail", "random")


def uses_v3_masking(cfgs_mask):
    return any(
        m.get("truncation", "tail") != "tail"
        or int(m.get("hide_time_steps", 0) or 0) > 0
        or (m.get("target_weight", "none") or "none") != "none"
        for m in cfgs_mask
    )


def build_mask_collator(cfgs_mask, dataset_fpcs, crop_size, patch_size, tubelet_size):
    """V-JEPA 2.1's collator unless a mask entry asks for one of the options."""
    cls = MaskCollatorV3 if uses_v3_masking(cfgs_mask) else MaskCollator
    return cls(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )


class MaskCollatorV3(MaskCollator):
    """The 2.1 collator with `_MaskGeneratorV3` in place of its generators."""

    def __init__(
        self, cfgs_mask, dataset_fpcs, crop_size=(224, 224), patch_size=(16, 16), tubelet_size=2
    ):
        super().__init__(
            cfgs_mask=cfgs_mask,
            dataset_fpcs=dataset_fpcs,
            crop_size=crop_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
        )
        for fpc in dataset_fpcs:
            self.mask_generators[fpc] = [
                _MaskGeneratorV3(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),
                    temporal_pred_mask_scale=m.get("temporal_scale"),
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                    full_complement=m.get("full_complement", False),
                    pred_full_complement=m.get("pred_full_complement", False),
                    inv_block=m.get("inv_block", False),
                    truncation=m.get("truncation", "tail"),
                    hide_time_steps=int(m.get("hide_time_steps", 0) or 0),
                )
                for m in cfgs_mask
            ]
        self.weight_settings = resolve_target_weights(cfgs_mask)
        self.patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
        self.tubelet_size = tubelet_size

    def __call__(self, batch):
        """As 2.1: one (batch, context ids, target ids) per clip length.  When
        a mask entry sets `target_weight`, a fourth item is added: per mask, a
        [batch, n_targets] tensor of weights, or an empty tensor for a mask
        without weights."""
        collations = super().__call__(batch)
        if not any(self.weight_settings):
            return collations
        weighted = []
        for collated, masks_enc, masks_pred in collations:
            clips = collated[0][0]  # [batch, 3, frames, height, width]
            per_mask = []
            for settings, ids_shown, ids_asked in zip(self.weight_settings, masks_enc, masks_pred):
                if settings is None:
                    per_mask.append(torch.empty(0))
                    continue
                rows = [
                    target_weights(
                        clips[b],
                        ids_shown[b].numpy(),
                        ids_asked[b].numpy(),
                        self.tubelet_size,
                        self.patch_size,
                        full=settings["full"],
                    )
                    for b in range(clips.shape[0])
                ]
                per_mask.append(torch.from_numpy(np.stack(rows)))
            weighted.append((collated, masks_enc, masks_pred, per_mask))
        return weighted


class _MaskGeneratorV3(_MaskGenerator):
    def __init__(self, *args, truncation="tail", hide_time_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        if truncation not in TRUNCATIONS:
            raise ValueError(f"mask truncation must be one of {TRUNCATIONS}, got {truncation!r}")
        if not 0 <= hide_time_steps < self.duration:
            raise ValueError(
                f"hide_time_steps must be between 0 and {self.duration - 1} "
                f"(the clip has {self.duration} time-steps), got {hide_time_steps}"
            )
        if hide_time_steps > 0 and (
            self.full_complement or self.pred_full_complement or self.inv_block
        ):
            raise ValueError(
                "hide_time_steps cannot be combined with full_complement, "
                "pred_full_complement or inv_block"
            )
        self.truncation = truncation
        self.hide_time_steps = hide_time_steps

    def _sample_frame_mask(self):
        """1 = visible, 0 = hidden: `hide_time_steps` whole time-steps hidden."""
        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        hidden = torch.randperm(self.duration)[: self.hide_time_steps]
        mask[hidden] = 0
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0
        return mask

    def _cut(self, ids, keep):
        """Cut a sorted list of token ids to `keep` entries, still sorted."""
        if self.truncation == "tail" or len(ids) <= keep:
            return ids[:keep]
        chosen = torch.randperm(len(ids))[:keep].sort().values
        return ids[chosen]

    def __call__(self, batch_size):
        """Same contract as the 2.1 generator: (context ids, target ids), each
        [batch, n] and sorted, or the reverse when `inv_block` is set."""
        # Advance the shared counter and draw the block size exactly as 2.1
        # does, whether or not it is used, so `MaskCollator.step()` on resume
        # keeps meaning the same thing.
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        masks_enc, masks_pred = [], []
        for _ in range(batch_size):
            while True:
                if self.hide_time_steps > 0:
                    mask = self._sample_frame_mask()
                else:
                    mask = torch.ones(
                        (self.duration, self.height, self.width), dtype=torch.int32
                    )
                    for _ in range(self.npred):
                        mask *= self._sample_block_mask(p_size)
                mask = mask.flatten()
                mask_e = torch.nonzero(mask).reshape(-1)
                if len(mask_e) > 0:
                    break
            masks_enc.append(mask_e)
            masks_pred.append(torch.argwhere(mask == 0).reshape(-1))

        keep_enc = min(len(m) for m in masks_enc)
        keep_pred = min(len(m) for m in masks_pred)
        if self.max_keep is not None:
            keep_enc = min(keep_enc, self.max_keep)
        masks_enc = [self._cut(m, keep_enc) for m in masks_enc]
        masks_pred = [self._cut(m, keep_pred) for m in masks_pred]

        n_tokens = int(self.duration * self.height * self.width)
        if self.full_complement:  # targets are everything the context is not
            masks_pred = [_complement(m, n_tokens) for m in masks_enc]
        elif self.pred_full_complement:
            masks_enc = [_complement(m, n_tokens) for m in masks_pred]

        masks_enc = torch.utils.data.default_collate(masks_enc)
        masks_pred = torch.utils.data.default_collate(masks_pred)
        if self.inv_block:
            return masks_pred, masks_enc
        return masks_enc, masks_pred


def _complement(ids, n_tokens):
    keep = torch.ones(n_tokens, dtype=torch.bool)
    keep[ids] = False
    return torch.nonzero(keep).reshape(-1).to(ids.dtype)
