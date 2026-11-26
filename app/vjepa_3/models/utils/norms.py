# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""V-JEPA 3: normalization-free replacements for LayerNorm (`norm_type` key).

WHAT THESE ARE
--------------
LayerNorm computes per-token statistics (mean and variance over channels) and
rescales by them.  DyT and Derf compute NO statistics.  Each is a fixed
element-wise S-shaped squashing function with a learnable input scale,
followed by the same per-channel affine (weight, bias) LayerNorm has:

    DyT    weight * tanh(alpha * x)         + bias      (Zhu et al., CVPR 2025)
    Derf   weight * erf (alpha * x + shift) + bias      (Chen et al., CVPR 2026,
                                                         arXiv 2512.10938)

`alpha` and `shift` are single scalars per layer, not per channel.  Initial
values follow the papers: alpha = 0.5, shift = 0, weight = 1, bias = 0.

WHAT IS AND IS NOT REPLACED
---------------------------
`norm_type` replaces the two pre-norms in every Block (norm1, norm2).
`norm_out_type` replaces the OUTPUT norms -- the encoder's four deep-supervision
`norms_block` and the predictor's `predictor_norm` -- and defaults to following
`norm_type`.  It is a separate key because the predictor's output is regressed
onto LayerNorm-standardised targets, an interaction specific to this objective.
(DyT's MAE experiment does replace the output norm before a regression head;
Derf reports no such case.  Ladder audit N-13.)

The `F.layer_norm` calls in train.py (target standardisation) are part of the
LOSS, not the architecture, and are never replaced.

THINGS THAT FOLLOW FROM THE SHAPES
----------------------------------
* Every parameter here is 1-D, so `init_opt`'s existing "bias or 1-D" rule
  puts all of them in the no-weight-decay group with no extra code.
* `_init_weights` in the ViT only touches Linear / LayerNorm / Conv, so the
  alpha = 0.5 set in __init__ survives `self.apply(self._init_weights)`.
* Under bf16 autocast, `alpha * x` multiplies an fp32 parameter by a bf16
  activation; type promotion makes the result fp32, so tanh/erf run in fp32.
  That matches LayerNorm, which autocast also runs in fp32.

NO WARM START
-------------
These change the function the network computes, so a V-JEPA 2.1 checkpoint
cannot initialise a `dyt`/`derf` model.  Those arms train from scratch.
"""

from functools import partial

import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    """DyT: weight * tanh(alpha * x) + bias, channels-last."""

    def __init__(self, dim, alpha_init=0.5):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return torch.tanh(self.alpha * x) * self.weight + self.bias

    def extra_repr(self):
        return f"dim={self.dim}"


class DynamicErf(nn.Module):
    """Derf: weight * erf(alpha * x + shift) + bias, channels-last."""

    def __init__(self, dim, alpha_init=0.5, shift_init=0.0):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.shift = nn.Parameter(torch.full((1,), float(shift_init)))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return torch.erf(self.alpha * x + self.shift) * self.weight + self.bias

    def extra_repr(self):
        return f"dim={self.dim}"


NORM_TYPES = ("layernorm", "dyt", "derf")


def build_norm_layer(norm_type, eps=1e-6):
    """Return a `dim -> nn.Module` constructor for the named norm.

    `eps` only applies to `layernorm`; 1e-6 is what every ViT factory in
    vision_transformer.py passes, so `norm_type: layernorm` is the same layer
    as leaving `norm_type` unset.
    """
    if norm_type == "layernorm":
        return partial(nn.LayerNorm, eps=eps)
    if norm_type == "dyt":
        return DynamicTanh
    if norm_type == "derf":
        return DynamicErf
    raise ValueError(
        f"Unknown norm_type {norm_type!r}. Known: {sorted(NORM_TYPES)} "
        "(or null to keep the factory's LayerNorm)."
    )
