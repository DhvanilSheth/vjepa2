# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""V-JEPA 3: feed-forward (FFN) variants, selected by the `ffn_type` config key.

WHAT THIS FILE IS
-----------------
The names and the small pieces that the FFN choice needs.  The two FFN module
classes themselves (`MLP`, `SwiGLUFFN`) stay in modules.py next to `Block`, and
`build_ffn` lives there too -- modules.py imports from here, so this file must
not import from modules.py.

THE VARIANTS
------------
    gelu    MLP:  fc2(GELU(fc1(x)))                 -- V-JEPA 2.1 default
    relu2   MLP:  fc2(ReLU(fc1(x))**2)              -- same shape, same params
    swiglu  GLU:  fc3(SiLU(fc1(x)) * fc2(x))        -- what `use_silu` selects
    geglu   GLU:  fc3(GELU(fc1(x)) * fc2(x))
    reglu   GLU:  fc3(ReLU(fc1(x)) * fc2(x))

`relu2` keeps the parameter count of `gelu` exactly, so that arm changes only
the nonlinearity.  The GLU family has three linears; with `wide_silu: true`
the hidden width is cut to 2/3 so the parameter count stays close to the MLP.

BACKWARD COMPATIBILITY
----------------------
`ffn_type: null` (the default) means "use the legacy `use_silu` switch", and
takes the unmodified V-JEPA 2.1 code path in `Block.__init__`.  So a config
that never mentions `ffn_type` builds a bitwise-identical model.

EVIDENCE (why these are flags, not defaults)
--------------------------------------------
None of these is an established win for vision transformers.  ViT-5
(arXiv 2602.08071) deliberately leaves SwiGLU out; "Bug or Feature^2"
(arXiv 2605.17659) reports squared activations do not beat GELU in vision.
They are here to be measured against the control arm, not assumed.
"""

import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    """relu(x) ** 2.  Zero for x <= 0, smooth at 0, unbounded above."""

    def forward(self, x):
        return F.relu(x).square()


# Gate nonlinearity for each member of the GLU family.  The gated branch is
# always `gate(fc1(x)) * fc2(x)`; only `gate` differs.
GLU_GATES = {
    "swiglu": F.silu,
    "geglu": F.gelu,
    "reglu": F.relu,
}

# Activation module for each plain-MLP variant.
MLP_ACTS = {
    "gelu": nn.GELU,
    "relu2": SquaredReLU,
}

FFN_TYPES = tuple(MLP_ACTS) + tuple(GLU_GATES)


def resolve_ffn_type(ffn_type, use_silu=False):
    """Validate `ffn_type` against the legacy `use_silu` switch.

    Returns `ffn_type` unchanged, INCLUDING `None`.  `None` is deliberately
    not turned into "gelu"/"swiglu" here: it is the signal for Block to take
    the untouched legacy branch, which is what keeps defaults bitwise 2.1.

    Raises ValueError for an unknown name, or when `use_silu: true` is
    combined with an `ffn_type` that is not `swiglu` -- two keys asking for
    different FFNs is a config mistake, and silently letting one win would
    mislabel an ablation arm.
    """
    if ffn_type is None:
        return None
    if ffn_type not in FFN_TYPES:
        raise ValueError(
            f"Unknown ffn_type {ffn_type!r}. Known: {sorted(FFN_TYPES)} "
            "(or null for the legacy use_silu switch)."
        )
    if use_silu and ffn_type != "swiglu":
        raise ValueError(
            f"use_silu=True conflicts with ffn_type={ffn_type!r}. "
            "Set use_silu: false (ffn_type alone selects the FFN), or use "
            "ffn_type: swiglu."
        )
    return ffn_type
