# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import drop_path

from app.vjepa_3.models.utils.string import apply_string


# -- V-JEPA 3: default RoPE base ("max wavelength").  V-JEPA 2.1 hardcoded
#    this inline at what is now :57.  See build_omega below for why it is
#    worth making configurable.
DEFAULT_ROPE_THETA = 10000.0


def build_omega(dim, theta=DEFAULT_ROPE_THETA, dtype=torch.float32, device=None):
    """Rotary frequency table for ONE axis: omega_i = theta ** (-i / (dim/2)).

    Args:
        dim:   the rotated width of this axis -- RoPEAttention.d_dim / h_dim /
               w_dim, i.e. 20 for head_dim=64.  Half that many frequencies.
        theta: the RoPE base.  10000.0 reproduces V-JEPA 2.1 exactly.

    Returns: (dim // 2,) tensor, frequencies descending from 1.0.

    WHY THIS IS CONFIGURABLE.  theta=10000 is inherited from language models,
    where sequences run to thousands of positions.  V-JEPA's grids are tiny by
    comparison -- 8 temporal positions at 16 frames / tubelet 2, 16 spatial at
    256px / patch 16 -- so most of the spectrum never turns.  Total rotation
    swept across the ENTIRE grid extent, for dim=20 (10 frequencies):

        temporal @16f  (8 pos): 7.00 2.79 1.11 .44 .18 .07 .03 .011 .004 .002
        temporal @64f (32 pos): 31.0 12.3 4.91 1.96 .78 .31 .12 .05 .02 .008
        spatial @256px (16 pos): 15.0 5.97 2.38 .95 .38 .15 .06 .02 .009 .004

    Channels sweeping under ~0.2 rad over the whole clip carry almost no
    positional signal: 6 of 10 temporal channels at 16 frames, 5 of 10 spatial.
    Lowering theta compresses the spectrum onto the range that actually exists.
    The STRING paper (Schenck et al., 2025) independently initialises with max
    wavelength 100 rather than 10,000, following RoPE-Mixed -- convergent
    evidence that the language-model default is mismatched here.

    Reproduce the table above with:
        python3 -c "n=10; om=[10000**(-i/n) for i in range(n)]; \\
                    print([round(o*7,3) for o in om])"
    """
    n = dim // 2
    omega = torch.arange(n, dtype=dtype, device=device)
    omega /= n
    return 1.0 / theta**omega


def _normalize_rope_theta(rope_theta):
    """Coerce the `rope_theta` config value to a validated (d, h, w) triple.

    Accepts a scalar (broadcast to all three axes) or a 3-sequence
    [theta_d, theta_h, theta_w].

    RAISES if theta_h != theta_w.  The two spatial axes must share a frequency
    spectrum or spatial isotropy breaks: with different bases, rotating an
    input 90 degrees would no longer correspond to relabelling the encoding,
    and the model would acquire a preferred image axis.  The temporal base is
    free to differ -- that asymmetry is the point (time and space have
    different extents), and it is the `[100, 10000, 10000]` ablation arm.
    """
    if isinstance(rope_theta, (int, float)):
        theta_d = theta_h = theta_w = float(rope_theta)
    else:
        values = list(rope_theta)
        if len(values) != 3:
            raise ValueError(
                f"rope_theta must be a scalar or a 3-sequence [d, h, w], "
                f"got {rope_theta!r}"
            )
        theta_d, theta_h, theta_w = (float(v) for v in values)

    if theta_h != theta_w:
        raise ValueError(
            f"rope_theta requires theta_h == theta_w for spatial isotropy, got "
            f"h={theta_h}, w={theta_w}. Differing spatial bases give the model "
            f"a preferred image axis."
        )
    for name, value in (("d", theta_d), ("h", theta_h), ("w", theta_w)):
        if value <= 1.0:
            raise ValueError(
                f"rope_theta[{name}] must be > 1.0 (it is the base of "
                f"theta**(-i/n)), got {value}"
            )
    return theta_d, theta_h, theta_w


def rotate_queries_or_keys(x, pos, n_registers, has_cls_first, omega=None):
    """Apply rotary position encoding to one axis-slice of q or k.

    omega: optional precomputed frequency table of shape (D // 2,).  When
        None, the exact V-JEPA 2.1 inline computation is used -- kept
        byte-identical rather than routed through build_omega, so that the
        default path is provably unchanged (see
        tests/models/test_vjepa3_parity.py).
    """
    B, num_heads, N, D = x.size()
    assert (
        D % 2 == 0
    ), "Embedding dimension must be a multiple of 2 for block matrix rotation"

    n_cls = 1 if has_cls_first else 0
    start_ctx = n_cls
    end_ctx = N - n_registers

    x_cls = x[..., :n_cls, :] if n_cls else None
    x_ctx = x[..., start_ctx:end_ctx, :]
    x_reg = x[..., end_ctx:, :] if n_registers > 0 else None

    if omega is None:
        # -- LEGACY PATH: byte-for-byte identical to V-JEPA 2.1.
        #    NOTE the dtype: under autocast(bfloat16) (train.py) `x` is bf16,
        #    so this frequency table is built IN BFLOAT16.  Computing it in
        #    fp32 "for accuracy" would silently break bitwise parity with 2.1.
        omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
        omega /= D / 2.0
        omega = 1.0 / 10000**omega
    else:
        # Cast at the point of use so the bf16 rounding pattern matches the
        # legacy path even when the table was precomputed in fp32.
        omega = omega.to(dtype=x.dtype, device=x.device)
    freq = torch.einsum("..., f -> ... f", pos, omega)

    emb_sin = freq.sin()
    emb_cos = freq.cos()

    emb_sin = emb_sin.repeat_interleave(2, dim=-1)
    emb_cos = emb_cos.repeat_interleave(2, dim=-1)

    y = x_ctx.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)

    out_ctx = (x_ctx * emb_cos) + (y * emb_sin)

    parts = []
    if n_cls:
        parts.append(x_cls)
    parts.append(out_ctx)
    if n_registers:
        parts.append(x_reg)
    out = torch.cat(parts, dim=-2)

    return out


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        drop=0.0,
        wide_silu=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        swiglu_hidden_features = hidden_features = hidden_features or in_features
        if wide_silu:
            swiglu_hidden_features = int(2 * hidden_features / 3)
            align_as = 8
            swiglu_hidden_features = (
                (swiglu_hidden_features + align_as - 1) // align_as * align_as
            )
        self.fc1 = nn.Linear(in_features, swiglu_hidden_features)
        self.fc2 = nn.Linear(in_features, swiglu_hidden_features)
        self.act = act_layer()
        self.fc3 = nn.Linear(swiglu_hidden_features, out_features)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1) * x2
        return self.fc3(hidden)


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
        rope_theta=DEFAULT_ROPE_THETA,
        rope_temporal_scale=1.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.n_registers = n_registers
        self.has_cls_first = has_cls_first
        self.interpolate_rope = interpolate_rope
        self.pretrained_patch_size = patch_size
        if patch_size == 14:
            self.pretrained_grid_size = int(252 / patch_size)
        elif patch_size == 16:
            self.pretrained_grid_size = int(256 / patch_size)

        # -- V-JEPA 3: per-axis RoPE base + adjustable temporal spacing.
        self.rope_theta = theta_d, theta_h, theta_w = _normalize_rope_theta(rope_theta)
        self.rope_temporal_scale = rope_temporal_scale

        # IMPORTANT -- why these are None at the default base, not buffers.
        # Building the frequency table in fp32 and casting to bf16 at use is
        # NOT bitwise equal to building it in bf16 throughout (`omega /= n` and
        # `1/theta**omega` round differently); they diverge in 4 of 10 channels
        # at head_dim=64.  A precomputed fp32 buffer would therefore silently
        # perturb numerics under autocast even at theta=10000.  So an axis left
        # at the default carries None and takes the legacy inline path inside
        # rotate_queries_or_keys, which is byte-identical to V-JEPA 2.1.  Only
        # an axis the config actually changes materialises a table.
        # Pinned by tests/models/test_vjepa3_rope.py.
        self.register_buffer(
            "omega_d",
            None if theta_d == DEFAULT_ROPE_THETA else build_omega(self.d_dim, theta_d),
            persistent=False,  # never enters state_dict -> can't break strict load
        )
        self.register_buffer(
            "omega_h",
            None if theta_h == DEFAULT_ROPE_THETA else build_omega(self.h_dim, theta_h),
            persistent=False,
        )
        self.register_buffer(
            "omega_w",
            None if theta_w == DEFAULT_ROPE_THETA else build_omega(self.w_dim, theta_w),
            persistent=False,
        )

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return 1.0 * frame_ids, 1.0 * height_ids, 1.0 * width_ids

    def forward(
        self,
        x,
        mask=None,
        T=None,
        H_patches=None,
        W_patches=None,
        return_attn=False,
        string_P=None,
    ):
        B, N, C = x.size()
        N_ctx = N - self.n_registers
        grid_depth = int(N_ctx // (self.grid_size * self.grid_size))

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # -- V-JEPA 3 STRING: learned orthogonal change of basis on q and k.
        #    Applied to ALL tokens (cls + context + registers) and BEFORE the
        #    d/h/w slicing below, so P can mix across the axis partitions --
        #    that mixing IS the generalisation.  `v` is untouched: STRING acts
        #    only on the pair that forms the attention score.  The outer P^T of
        #    P^T D(r) P cancels inside the inner product, which is why only
        #    this one multiplication is needed.  See string.py for the
        #    derivation.  string_P is None unless use_string is enabled.
        if string_P is not None:
            q = apply_string(q, string_P)
            k = apply_string(k, string_P)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)
        else:
            if T is None or H_patches is None or W_patches is None:
                mask = torch.arange(
                    int(grid_depth * self.grid_size * self.grid_size), device=x.device
                )
            else:
                mask = torch.arange(int(T * H_patches * W_patches), device=x.device)
            d_mask, h_mask, w_mask = self.separate_positions(mask, H_patches, W_patches)

        # -- V-JEPA 3: adjustable temporal spacing (VideoRoPE's ATS).
        #    Scales how fast the temporal index advances relative to spatial.
        #    Default 1.0 is a dead branch -> bitwise no-op.  Values < 1.0
        #    compress temporal positions, which is the cheapest mitigation for
        #    the 16f -> 64f shift between the pretrain and cooldown recipes
        #    (8 vs 32 temporal positions through the same frequency table).
        #    Applied BEFORE interpolate_rope, which only touches h/w.
        if self.rope_temporal_scale != 1.0:
            d_mask = d_mask * self.rope_temporal_scale

        if self.interpolate_rope:
            if H_patches is None:
                H_patches = int(self.grid_size)
            if W_patches is None:
                W_patches = int(self.grid_size)
            h_mask = h_mask * (self.pretrained_grid_size - 1) / (H_patches - 1)
            w_mask = w_mask * (self.pretrained_grid_size - 1) / (W_patches - 1)

        s = 0
        qd = rotate_queries_or_keys(
            q[..., s : s + self.d_dim],
            pos=d_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
            omega=self.omega_d,
        )
        kd = rotate_queries_or_keys(
            k[..., s : s + self.d_dim],
            pos=d_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
            omega=self.omega_d,
        )
        s += self.d_dim
        qh = rotate_queries_or_keys(
            q[..., s : s + self.h_dim],
            pos=h_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
            omega=self.omega_h,
        )
        kh = rotate_queries_or_keys(
            k[..., s : s + self.h_dim],
            pos=h_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
            omega=self.omega_h,
        )
        s += self.h_dim
        qw = rotate_queries_or_keys(
            q[..., s : s + self.w_dim],
            pos=w_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
            omega=self.omega_w,
        )
        kw = rotate_queries_or_keys(
            k[..., s : s + self.w_dim],
            pos=w_mask,
            n_registers=self.n_registers,
            has_cls_first=self.has_cls_first,
            omega=self.omega_w,
        )
        s += self.w_dim

        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        else:
            return x, None


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.is_causal = is_causal

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        patch_size=16,
        # -- V-JEPA 3.  NOTE these are named params, NOT left to **kwargs:
        #    this __init__ ends in **kwargs, so a misspelled or unthreaded key
        #    is swallowed silently with no error.  That is not hypothetical --
        #    `grid_depth` is passed to Block from vision_transformer.py and
        #    predictor.py today and vanishes exactly this way.
        rope_theta=DEFAULT_ROPE_THETA,
        rope_temporal_scale=1.0,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_rope = use_rope
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
                n_registers=n_registers,
                has_cls_first=has_cls_first,
                rope_theta=rope_theta,
                rope_temporal_scale=rope_temporal_scale,
                interpolate_rope=interpolate_rope,
                patch_size=patch_size,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = SwiGLUFFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                wide_silu=wide_silu,
                drop=drop,
            )
        else:
            self.mlp = MLP(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

    def forward(
        self,
        x,
        mask=None,
        T=None,
        H_patches=None,
        W_patches=None,
        return_attn=False,
        mode="video",
        string_P=None,
    ):
        # -- V-JEPA 3: string_P arrives at FORWARD time, not construction time.
        #    Unlike rope_theta (a Block.__init__ param), the orthogonal basis is
        #    a live tensor owned by the model and recomputed each step, so the
        #    Block only forwards it.  Ignored on the non-RoPE branch, which has
        #    no positional encoding to compose with.
        if self.use_rope:
            y, attn = self.attn(
                self.norm1(x),
                mask=mask,
                T=T,
                H_patches=H_patches,
                W_patches=W_patches,
                return_attn=return_attn,
                string_P=string_P,
            )
        else:
            y = self.attn(self.norm1(x))
            attn = None
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        else:
            return x, None


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, use_sdpa=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = (
            self.q(q)
            .reshape(B, n, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        B, N, C = x.shape
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            xattn = xattn.softmax(dim=-1)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        return q


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


class Lambda_LinearWarmupHold:
    """
    Linear warmup from 0 to lambda_value between [start_iter, end_iter],
    0 before start_iter and constant (=lambda_value) from end_iter onwards.
    """

    def __init__(
        self, lambda_value: float, start_iter: int = 15_000, end_iter: int = 30_000
    ):
        assert end_iter > start_iter, "end_iter must be > start_iter"
        self.lambda_value = float(lambda_value)
        self.start = int(start_iter)
        self.end = int(end_iter)
        self.span = self.end - self.start

    def value(self, global_iter: int) -> float:
        if global_iter < self.start:
            return 0.0
        if global_iter >= self.end:
            return self.lambda_value
        alpha = (global_iter - self.start) / self.span
        return self.lambda_value * alpha
