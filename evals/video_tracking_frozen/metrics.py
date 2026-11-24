# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DAVIS video-object-segmentation metrics: J (region) and F (boundary).

Re-implemented from the definitions in Perazzi et al. (CVPR 2016) and the
structure of the official `davis2017-evaluation` toolkit, so the probe needs
no extra package.  NOT yet cross-checked against that toolkit's numbers on
real predictions; the unit tests cover closed-form cases only.

  J  intersection-over-union of the predicted and ground-truth mask.
  F  F-measure between the two masks' boundaries, where a boundary pixel
     counts as matched if the other boundary passes within `bound_th` of the
     image diagonal (0.008 -> 8 px at 480p).
"""

import numpy as np
import torch
import torch.nn.functional as F


def region_similarity(pred, gt):
    """J: IoU of two boolean [H, W] masks.  Two empty masks score 1."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def seg2bmap(seg):
    """Boundary map of a boolean mask: a pixel is on the boundary if it
    differs from its east, south or south-east neighbour."""
    seg = np.asarray(seg, dtype=bool)
    e, s, se = np.zeros_like(seg), np.zeros_like(seg), np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = (seg ^ e) | (seg ^ s) | (seg ^ se)
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def _dilate_disk(mask, radius):
    """Binary dilation of a boolean [H, W] array by a disk of `radius` px."""
    radius = int(radius)
    if radius < 1:
        return mask
    axis = torch.arange(-radius, radius + 1)
    disk = (axis[:, None] ** 2 + axis[None, :] ** 2 <= radius**2).float()
    hit = F.conv2d(
        torch.from_numpy(mask.astype(np.float32))[None, None],
        disk[None, None],
        padding=radius,
    )
    return (hit[0, 0] > 0).numpy()


def boundary_f_measure(pred, gt, bound_th=0.008):
    """F: boundary precision/recall F-measure of two boolean [H, W] masks."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(pred.shape))
    pred_b, gt_b = seg2bmap(pred), seg2bmap(gt)
    n_pred, n_gt = pred_b.sum(), gt_b.sum()
    if n_pred == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_pred > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_pred == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = (pred_b & _dilate_disk(gt_b, bound_pix)).sum() / n_pred
        recall = (gt_b & _dilate_disk(pred_b, bound_pix)).sum() / n_gt
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
