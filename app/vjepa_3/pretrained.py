# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""V-JEPA 3: starting from a released V-JEPA 2.1 encoder.

WHAT CAN BE CARRIED OVER
------------------------
Meta's released files (README, "V-JEPA 2.1 Pretrained Checkpoints") hold an
encoder and a predictor.  For ViT-B and ViT-L the model was DISTILLED from
ViT-G, and the predictor in the file is the distillation head: it outputs the
teacher's 1664-wide features at one level.  The pre-training loop here
predicts the encoder's own width at four levels, so that predictor does not
fit, and the distillation loop is not in this repository.

So only the ENCODER is loaded.  Its architecture is identical to the one the
v3 configs build with default flags (158 of 158 tensors, same shapes;
tests/models/test_vjepa3_pretrained.py), which is why this works for every
arm that keeps the encoder's parameters: RoPE base changes, STRING, and
the regularisers.  It does not work for a different FFN or norm.

The predictor starts untrained.  See `optimization.freeze_encoder_iters` and
`optimization.encoder_lr_scale` in train.py for the two guards against an
untrained predictor damaging a trained encoder.

NOT VERIFIED: no released file has been opened by this code yet.  The key
names below come from `src/hub/backbones.py`, which is Meta's own loader for
the same files.
"""

import torch

# Released V-JEPA 2.1 models: file name -> (factory name, checkpoint key).
# Taken from ARCH_NAME_MAP and the `vjepa2_1_*` functions in
# src/hub/backbones.py.
RELEASED_2_1 = {
    "vjepa2_1_vitb_dist_vitG_384": ("vit_base", "ema_encoder"),
    "vjepa2_1_vitl_dist_vitG_384": ("vit_large", "ema_encoder"),
    "vjepa2_1_vitg_384": ("vit_giant_xformers", "target_encoder"),
    "vjepa2_1_vitG_384": ("vit_gigantic_xformers", "target_encoder"),
}

_PREFIXES = ("module.", "backbone.")


def released_2_1_pretrain_config(model_name):
    """The part of a pre-training config the probes need to rebuild a released
    V-JEPA 2.1 encoder: the same arguments `_make_vjepa2_1_model` passes in
    src/hub/backbones.py, in this repo's config vocabulary.
    """
    known = sorted({arch for arch, _ in RELEASED_2_1.values()})
    if model_name not in known:
        raise ValueError(f"no released V-JEPA 2.1 model is a {model_name!r}; known: {known}")
    return {
        "model": {
            "model_name": model_name,
            "use_rope": True,
            "interpolate_rope": True,
            "img_temporal_dim_size": 1,
            "uniform_power": False,
            "modality_embedding": True,
            "use_silu": False,
            "wide_silu": True,
        },
        "data": {"patch_size": 16, "tubelet_size": 2},
        "meta": {"use_sdpa": True},
    }


def clean_encoder_state_dict(state_dict):
    """Strip the `module.` / `backbone.` wrappers from every key."""
    cleaned = {}
    for key, value in state_dict.items():
        stripped = True
        while stripped:
            stripped = False
            for prefix in _PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    stripped = True
        cleaned[key] = value
    return cleaned


def read_encoder_state_dict(path, checkpoint_key="ema_encoder"):
    """Read one encoder's weights from a checkpoint file.

    `weights_only=True`: a downloaded file is untrusted, and a state dict
    needs nothing but tensors.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint_key not in checkpoint:
        raise KeyError(
            f"{path} has no {checkpoint_key!r} entry; it has {sorted(checkpoint)}. "
            "Released ViT-B/L use 'ema_encoder', ViT-g/G use 'target_encoder'; "
            "this repo's own checkpoints use 'target_encoder' or 'encoder'."
        )
    return clean_encoder_state_dict(checkpoint[checkpoint_key])


def load_pretrained_encoder(encoder, path, checkpoint_key="ema_encoder"):
    """Load a released (or any v3) encoder into `encoder`, strictly.

    `encoder` may be the bare ViT or train.py's `MultiSeqWrapper` around it.
    strict=True on purpose: a silent partial load would train a half-random
    model under a "warm start" label.
    """
    backbone = getattr(encoder, "backbone", encoder)
    backbone.load_state_dict(read_encoder_state_dict(path, checkpoint_key), strict=True)
    return encoder
