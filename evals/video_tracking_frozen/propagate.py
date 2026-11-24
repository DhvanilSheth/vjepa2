# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Non-parametric label propagation (Jabri et al., 2020), as used by DINO and
by V-JEPA 2.1 (paper Section C.3) for video object segmentation.

Nothing is trained.  The first frame's object masks are turned into per-patch
labels; every later frame's patches copy labels from the patches they are
most similar to (cosine similarity of frozen features) in the first frame and
in a few preceding frames.  A mask therefore survives only as far as the
features of the same object stay similar across frames and stay different
from the background -- which is what the probe measures.
"""

import torch
import torch.nn.functional as F


def neighborhood_mask(h, w, size, shape="circle", device=None):
    """[h*w, h*w] bool: may target cell i copy from source cell j?

    True when j lies within `size` cells of i (Euclidean for "circle",
    Chebyshev for "square").  `size` None or <= 0 means no restriction and
    returns None.
    """
    if size is None or size <= 0:
        return None
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    ys, xs = ys.flatten(), xs.flatten()
    dy = (ys[:, None] - ys[None, :]).abs()
    dx = (xs[:, None] - xs[None, :]).abs()
    if shape == "circle":
        return dy**2 + dx**2 <= size**2
    if shape == "square":
        return (dy <= size) & (dx <= size)
    raise ValueError(f"neighborhood shape must be 'circle' or 'square', got {shape!r}")


def downsample_labels(onehot, patch_size):
    """[C, H, W] one-hot pixel labels -> [C, h, w] per-patch label fractions.

    Area averaging, not nearest sampling: an object smaller than a patch keeps
    a non-zero share of its patch instead of vanishing when the sampled pixel
    happens to miss it.
    """
    return F.avg_pool2d(onehot[None].float(), patch_size)[0]


def propagate_step(feat_tar, feats_src, labels_src, mask, topk, temperature):
    """Labels for one target frame.

    feat_tar   [hw, D]      L2-normalised target features
    feats_src  [S, hw, D]   L2-normalised features of the S source frames
    labels_src [S, C, hw]   their (soft) labels
    mask       [hw, hw] bool or None, see `neighborhood_mask`
    returns    [C, hw]
    """
    n_src, hw, dim = feats_src.shape
    n_classes = labels_src.shape[1]
    sim = feat_tar @ feats_src.reshape(n_src * hw, dim).T  # [hw, S*hw]
    if mask is not None:
        sim = sim.masked_fill(~mask.repeat(1, n_src), float("-inf"))
    values, index = sim.topk(min(topk, sim.shape[1]), dim=1)  # [hw, k]
    # softmax over the k kept neighbours == DINO's exp(sim / T), zero the
    # rest, renormalise.  A -inf (masked) neighbour gets weight 0.
    weights = torch.softmax(values / temperature, dim=1).nan_to_num(0.0)
    labels = labels_src.permute(1, 0, 2).reshape(n_classes, n_src * hw)
    return (labels[:, index] * weights[None]).sum(dim=-1)


def propagate_video(
    feats,
    first_labels,
    n_context=15,
    size=12,
    shape="circle",
    topk=5,
    temperature=0.2,
):
    """Propagate first-frame labels through a video.

    feats        [N, h, w, D] frozen features, one map per frame
    first_labels [C, h, w]    per-patch label fractions of frame 0
                              (channel 0 = background)
    returns      [N, C, h, w] soft labels; entry 0 is `first_labels`

    Sources for frame t are frame 0 (ground truth) plus the `n_context`
    frames before t (their own propagated labels).  Defaults are the
    configuration V-JEPA 2.1 reports for DAVIS.
    """
    n_frames, h, w, dim = feats.shape
    n_classes = first_labels.shape[0]
    f = F.normalize(feats.reshape(n_frames, h * w, dim).float(), dim=-1)
    mask = neighborhood_mask(h, w, size, shape, device=feats.device)
    labels = [first_labels.reshape(n_classes, h * w).float().to(feats.device)]
    for t in range(1, n_frames):
        context = [0] + list(range(max(1, t - n_context), t))
        labels.append(
            propagate_step(
                f[t],
                f[context],
                torch.stack([labels[i] for i in context]),
                mask,
                topk,
                temperature,
            )
        )
    return torch.stack(labels).reshape(n_frames, n_classes, h, w)


def labels_to_mask(soft, size, normalize=True):
    """[C, h, w] soft labels -> [H, W] integer mask at pixel resolution.

    Bilinear upsampling, then argmax over classes.  normalize=True rescales
    each class to [0, 1] first (DINO's `norm_mask`), which keeps a small or
    faint object from being out-voted everywhere by the background.
    """
    up = F.interpolate(soft[None].float(), size=size, mode="bilinear", align_corners=False)[0]
    if normalize:
        low = up.amin(dim=(1, 2), keepdim=True)
        high = up.amax(dim=(1, 2), keepdim=True)
        scaled = (up - low) / (high - low).clamp_min(1e-12)
        up = torch.where(high > 0, scaled, up)
    return up.argmax(dim=0)
