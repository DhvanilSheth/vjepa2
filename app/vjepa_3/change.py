# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""V-JEPA 3: how much of a hidden patch could simply be copied, and a loss
weight built from it.

OFF BY DEFAULT.  Nothing here runs unless a `mask:` entry sets
`target_weight: unexplained`.

THE GAP THIS ADDRESSES
----------------------
V-JEPA hides the same region in every frame.  Its authors tried masks that
cover less of the clip in time, found them worse, called the task trivial, and
kept full coverage (Bardes et al., 2024, section E.4).  Trivial was treated as
a property of the whole mask.  It is a property of each patch:

  - a patch that looks the same in a frame the model was shown can be copied;
  - a patch that looks different (something moved there) cannot.

Full coverage removes the first kind by never letting the model see any place
at another time, which removes the second kind too.  And every hidden patch
counts the same in the loss.  With `loss_exp: 1.0` the slope of the loss is
+1 or -1 whatever the size of the error, so a copyable patch pushes on the
model exactly as hard as a hard one.

Measured on this project's training videos (scripts/measure_change.py): at a
gap of 0.25 s, 57% of patches are explained by the one motion most of the
frame shares, and 80% at a gap of one frame.  A mask that hides whole moments
spends that share of its targets on patches that can be copied.

WHAT THIS MODULE DOES
---------------------
For a mask that hides whole time-steps, each hidden patch gets a weight in
[0, 1]: how much the shown time-steps fail to explain it.

    for each shown time-step t':
        undo the shift most of the frame shares between t and t'   (shared_shift)
        measure what still differs, patch by patch                 (patch_change)
        turn it into a number in [0, 1]                            (unexplained_weight)
    weight = the smallest of those numbers

The smallest, because one shown time-step that explains a patch is enough to
make it copyable.  A patch that was outside the picture at t' cannot be copied
from t', so it counts as 1 there.

The loss on that mask becomes  sum(w * |error|) / sum(w)  (weighted_error).

LIMITS
------
- One shift does not describe zoom, rotation or depth.  In such clips little
  is explained and the weights stay near 1, which is today's behaviour.
- "Unexplained" is not "an object".  On DAVIS the annotated objects are 30% of
  the patches and receive about half of the weight.
- A token covers `tubelet_size` frames; only the first frame of each is
  compared.
- The scale `full` (the change that earns weight 1) was read off DAVIS, which
  is also the test set.

The signal functions work on numpy arrays, one frame pair at a time.  They run
in the data-loading workers, and scripts/measure_change.py uses the same ones,
so training weights and reported measurements cannot drift apart.
"""

import numpy as np
import torch

TARGET_WEIGHTS = ("none", "unexplained")
DEFAULT_FULL = 0.064

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LUMA = (0.299, 0.587, 0.114)  # ITU-R 601, what PIL's "L" mode uses


# ---------------------------------------------------------------------------
# The signal: two grayscale frames in, one number per patch out.
# ---------------------------------------------------------------------------


def _pool(frame, cell):
    """[H, W] -> [H/cell, W/cell], the mean of each cell."""
    h, w = frame.shape
    return frame.reshape(h // cell, cell, w // cell, cell).mean((1, 3))


def _per_patch(pooled, cells):
    """[Gh*cells, Gw*cells] -> [Gh, Gw, cells, cells], each patch's mean removed."""
    gh, gw = pooled.shape[0] // cells, pooled.shape[1] // cells
    patches = pooled.reshape(gh, cells, gw, cells).transpose(0, 2, 1, 3)
    return patches - patches.mean((2, 3), keepdims=True)


def _shifted(frame, dy, dx):
    """`frame` moved by (dy, dx) pixels; the uncovered border repeats the edge."""
    pad = max(abs(dy), abs(dx))
    if pad == 0:
        return frame
    padded = np.pad(frame, pad, mode="edge")
    h, w = frame.shape
    return padded[pad - dy : pad - dy + h, pad - dx : pad - dx + w]


def patch_change(a, b, patch=16, cell=4, tol=1, gain=(0.8, 1.25)):
    """How much each patch differs between two frames.

    a, b: [H, W] grayscale in [0, 1], H and W multiples of `patch`.
    Returns [H/patch, W/patch]; 0 means identical.  Units are grey levels on
    the 0-1 scale, so 0.01 is about 2.5 of 255.

    Three steps, each removing one way a patch can differ in pixels while
    showing the same thing:
      low-pass      average `cell` x `cell` pixels: sensor and compression noise;
      brightness    subtract each patch's mean and fit one gain per patch;
      small shift   take the smallest difference over shifts of up to `tol`
                    pixels: camera shake.  `tol` is also the smallest motion
                    that counts as change.
    """
    cells = patch // cell
    pa = _per_patch(_pool(a, cell), cells)
    energy = np.maximum((pa * pa).sum((2, 3), keepdims=True), 1e-8)
    best = None
    for dy in range(-tol, tol + 1):
        for dx in range(-tol, tol + 1):
            pb = _per_patch(_pool(_shifted(b, dy, dx), cell), cells)
            g = np.clip((pa * pb).sum((2, 3), keepdims=True) / energy, *gain)
            d = np.abs(pb - g * pa).mean((2, 3))
            best = d if best is None else np.minimum(best, d)
    return best


def shared_shift(a, b, cell=4, reach=12):
    """The (dy, dx) in pixels by which `b` must be moved to lie on `a`.

    Coarse, then fine.  Coarse: every shift of up to `reach` cells is tried on
    the low-pass frames.  Fine: every shift within one cell of that answer is
    tried on the pixels, because a leftover of a few pixels would itself read
    as change.  Each time the shift with the smallest MEDIAN difference wins,
    so the larger part of the frame decides: a moving object does not, and
    neither does the strip of repeated edge pixels a shift leaves behind
    (under a fifth of the frame at the default `reach` and 256 pixels).  Zero
    wins ties.
    """

    def best_shift(x, y, candidates):
        best, winner = None, (0, 0)
        for dy, dx in sorted(candidates, key=lambda s: abs(s[0]) + abs(s[1])):
            score = np.median(np.abs(x - _shifted(y, dy, dx)))
            if best is None or score < best:
                best, winner = score, (dy, dx)
        return winner

    la, lb = _pool(a, cell), _pool(b, cell)
    span = range(-reach, reach + 1)
    cy, cx = best_shift(la - la.mean(), lb - lb.mean(), [(dy, dx) for dy in span for dx in span])
    fine = range(-(cell - 1), cell)
    return best_shift(
        a - a.mean(), b - b.mean(), [(cy * cell + dy, cx * cell + dx) for dy in fine for dx in fine]
    )


def covered_patches(dy, dx, size=256, patch=16):
    """[G, G] True for patches that lie fully inside the frame after it was
    moved by (dy, dx): the rest contain repeated edge pixels.  `size` is the
    frame's side in pixels, or (height, width)."""
    height, width = (size, size) if isinstance(size, int) else size
    inside = np.zeros((height, width), bool)
    inside[max(0, dy) : height + min(0, dy), max(0, dx) : width + min(0, dx)] = True
    return inside.reshape(height // patch, patch, width // patch, patch).all((1, 3))


def unexplained_weight(change, full):
    """Weight in [0, 1] for each patch: 0 where nothing changed, rising in
    proportion, 1 from a change of `full` upwards."""
    return np.clip(np.asarray(change) / full, 0.0, 1.0)


# ---------------------------------------------------------------------------
# From a clip and its mask to one weight per hidden token.
# ---------------------------------------------------------------------------


def gray_time_steps(clip, tubelet_size):
    """[3, T, H, W] normalised clip -> [T / tubelet_size, H, W] grayscale in
    [0, 1], float32 numpy: the first frame of every time-step."""
    frames = clip[:, ::tubelet_size].float()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
    luma = torch.tensor(LUMA).view(3, 1, 1, 1)
    return ((frames * std + mean) * luma).sum(0).clamp(0.0, 1.0).numpy()


def shown_time_steps(ids_shown, tokens_per_step):
    """The time-steps a list of visible token ids covers, each completely.

    The weight compares whole frames, so it needs to know which frames the
    model sees.  A time-step that is only partly visible has no such answer,
    and is refused.
    """
    steps, counts = np.unique(np.asarray(ids_shown) // tokens_per_step, return_counts=True)
    if not (counts == tokens_per_step).all():
        raise ValueError(
            "target_weight needs a mask that shows whole time-steps (hide_time_steps > 0); "
            f"this one shows {counts.tolist()} of {tokens_per_step} tokens in time-steps "
            f"{steps.tolist()}"
        )
    return steps.tolist()


def unexplained_map(frames, step, shown, full, patch=16, tol=1):
    """[Gh, Gw] weight of every patch of time-step `step`: how much the
    time-steps in `shown` fail to explain it.  frames: [T', H, W] grayscale."""
    a = frames[step]
    weight = np.ones((a.shape[0] // patch, a.shape[1] // patch), np.float32)
    for other in shown:
        dy, dx = shared_shift(a, frames[other])
        from_other = unexplained_weight(
            patch_change(a, _shifted(frames[other], dy, dx), patch=patch, tol=tol), full
        )
        from_other[~covered_patches(dy, dx, a.shape, patch)] = 1.0  # was out of view
        weight = np.minimum(weight, from_other)
    return weight


def target_weights(clip, ids_shown, ids_asked, tubelet_size, patch, full=DEFAULT_FULL, tol=1):
    """One weight per asked token of one clip, in the order of `ids_asked`.

    clip: [3, T, H, W] as the model receives it.  Token id = time-step *
    (tokens per time-step) + row * (tokens per row) + column, as in
    `RoPEAttention.separate_positions`.
    """
    frames = gray_time_steps(clip, tubelet_size)
    per_row = frames.shape[2] // patch
    per_step = (frames.shape[1] // patch) * per_row
    shown = shown_time_steps(ids_shown, per_step)
    ids_asked = np.asarray(ids_asked)
    weights = np.empty(len(ids_asked), np.float32)
    for step in np.unique(ids_asked // per_step):
        here = ids_asked // per_step == step
        grid = unexplained_map(frames, int(step), shown, full, patch, tol)
        weights[here] = grid.ravel()[ids_asked[here] % per_step]
    return weights


def resolve_target_weights(cfgs_mask, has_cls_first=False):
    """Per mask entry, None or the weight settings {"full": ...}; raises on a
    setting that cannot work rather than training something else."""
    resolved = []
    for index, entry in enumerate(cfgs_mask):
        kind = entry.get("target_weight", "none") or "none"
        if kind not in TARGET_WEIGHTS:
            raise ValueError(
                f"mask.{index}.target_weight must be one of {TARGET_WEIGHTS}, got {kind!r}"
            )
        if kind == "none":
            resolved.append(None)
            continue
        if int(entry.get("hide_time_steps", 0) or 0) <= 0:
            raise ValueError(
                f"mask.{index}.target_weight: {kind} needs hide_time_steps > 0 on the same "
                "mask: the weight compares whole frames the model is shown with whole "
                "frames it is asked for"
            )
        if has_cls_first:
            raise ValueError("target_weight is not implemented for models with a class token")
        full = float(entry.get("target_weight_full", DEFAULT_FULL))
        if not full > 0:
            raise ValueError(f"mask.{index}.target_weight_full must be positive, got {full}")
        resolved.append({"full": full})
    return resolved


# ---------------------------------------------------------------------------
# The loss term.
# ---------------------------------------------------------------------------


def weighted_error(prediction, target, weights, exponent, floor=0.01):
    """sum(w * |prediction - target| ** exponent) / (sum(w) * D) / exponent.

    prediction, target: [B, N, D]; weights: [B, N].  With all weights 1 this is
    the plain `mean(|.| ** exponent) / exponent`.

    The sums are taken in float32: under float16 autocast a sum over a batch
    of tokens passes float16's largest number.  `floor` keeps the denominator
    at no less than that share of the tokens, so a batch in which nothing
    moves gives a small loss instead of a division by nearly zero.
    """
    error = torch.abs(prediction - target).float() ** exponent
    weights = weights.float()
    total = torch.clamp(weights.sum(), min=floor * weights.numel())
    return (error * weights.unsqueeze(-1)).sum() / (total * error.shape[-1]) / exponent
