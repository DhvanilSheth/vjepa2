# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np


# FIXED, ADDITIVE positional embedding -- the OTHER branch from RoPE.
# In vision_transformer.py / predictor.py, `use_rope` picks exactly one of:
#   use_rope=True  -> position never touches the token stream; it's injected
#                     per-layer inside attention as a Q/K rotation
#                     (see rotate_queries_or_keys in modules.py)
#   use_rope=False -> position is computed ONCE here, stored as a non-trainable
#                     buffer, and simply added to the tokens before the first
#                     transformer block (x = x + pos_embed)
# These are mutually exclusive, not combined -- when use_rope=True this whole
# file is dead code for the backbone. It's kept for legacy/ablation configs
# (e.g. V-JEPA v1 style) and for the eval-time multi-clip aggregator, which
# always uses get_1d_sincos_pos_embed regardless of the backbone's PE choice.
#
# All three "get_Nd_sincos" functions below are just: split the embedding
# dimension into one slice per axis, run get_1d_sincos_pos_embed_from_grid
# on each axis independently, concatenate the slices back together. That's
# the additive analogue of what RoPE's Attention.forward does by splitting
# head_dim into d_dim/h_dim/w_dim and rotating each slice separately.

def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """
    grid_size: int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_h, grid_d, grid_w = np.meshgrid(
        grid_h, grid_d, grid_w
    )  # order of meshgrid is very important for indexing as [d,h,w]
    # each of grid_d/grid_h/grid_w is now (T,H,W) -- for every one of the
    # T*H*W flattened token positions, its own d/h/w grid coordinate

    if not uniform_power:
        # Default split: temporal gets HALF the embedding dim, height and
        # width share the other half (a quarter each). Note this is the
        # opposite bias from RoPE's Attention module, which splits head_dim
        # into three EQUAL thirds (d_dim == h_dim == w_dim) -- same repo,
        # two different design choices for how much "capacity" time gets,
        # because these are two independent, never-combined code paths.
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        # uniform_power: give all three axes an (approximately) equal share
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    # concat -> disjoint dims per axis (temporal | height | width), same
    # "each axis owns its own channels" idea as RoPE, just added instead of rotated
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]  # trim any rounding overshoot from uniform_power
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    returns:
        pos_embed: [grid_size*grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size*grid_size, embed_dim] (w/ cls_token)
    """
    # Same idea as the 3D version but for a single still image (no time
    # axis) -- this is what backs images fed through the tubelet=1 patch
    # embed path when use_rope=False.
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # order of meshgrid is very important for indexing as [h, w]

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D) -- height half, width half
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    # The only one of the three still called unconditionally in this repo --
    # not for the backbone, but in the eval-time multi-clip aggregator
    # (vit_encoder_multiclip.py), to tag each pooled clip embedding with
    # which temporal segment of the full video it came from. Independent of
    # whatever PE the frozen backbone used to produce that embedding.
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# The actual sin/cos math -- every function above is just bookkeeping
# (splitting dims, building grids) around this one core.
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    # Same frequency formula as RoPE's omega in modules.py:
    #   omega_i = 1 / 10000^(2i/D), i = 0..D/2-1
    # i.e. high index -> low frequency -> slow-varying code -> useful for
    # telling far-apart positions apart; low index -> high frequency ->
    # useful for telling nearby positions apart. RoPE uses this to build a
    # rotation angle per pair of channels; here it's used to build a fixed
    # sin/cos VECTOR that gets added to the token instead.
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product: position * frequency, for every (position, freq) pair

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    # Concat (not interleave) sin and cos halves into one vector per position --
    # this concatenated vector IS the positional embedding that gets added to
    # the token's content vector.
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
