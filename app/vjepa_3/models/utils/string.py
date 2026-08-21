# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Cayley-STRING: a learned orthogonal basis for rotary position encoding.

Reference: Schenck et al., "STRING: Separable Translationally Invariant
Position Encodings" (2025).

WHY -- what this replaces
-------------------------
V-JEPA 2.1's RoPEAttention hard-partitions every attention head: with
head_dim=64 it sets d_dim = h_dim = w_dim = 20 (modules.py:253-255), so dims
0-19 encode time, 20-39 height, 40-59 width, and the last 4 are unrotated.
That partition is an arbitrary design choice -- nothing says the three axes
should not share subspaces.  STRING relaxes it into a LEARNED basis while
provably keeping translation invariance.

THE MATH -- why this is one matmul, not three
---------------------------------------------
Cayley-STRING is  STRING(q, r) = P^T D(r) P q,  where D(r) is the ordinary
rotary rotation at 3-D position r = (t, y, x) and P is orthogonal.  Attention
consumes only inner products, and the outer P^T cancels against its partner:

    <P^T D(r) P q , P^T D(r') P k>
        = q^T P^T D(r)^T (P P^T) D(r') P k
        = q^T P^T D(r)^T D(r') P k            (P orthogonal => P P^T = I)
        = q^T P^T D(r'-r) P k

The score depends only on r' - r, so the encoding is still translation
invariant -- and the outer P^T never needs to be computed.  It therefore
suffices to LEFT-MULTIPLY q and k BY P BEFORE the existing rotation:

    R(r) = D(r) P

That is the whole implementation: one [D, D] matmul on q and on k, before the
d/h/w slicing.  `v` is never transformed -- STRING acts only on the pair that
forms the score.

WHY INVARIANCE SURVIVES THIS CODEBASE'S BLOCK STRUCTURE
-------------------------------------------------------
Here D(r) is block-diagonal across the axis slices:

    D(r) = blockdiag( D_d(t), D_h(y), D_w(x), I_4 )

Each rotary block satisfies D(p)^T D(p') = D(p' - p), and the 4 leftover
unrotated dims form an identity block, which is the rotation by zero and so
satisfies it trivially.  A block-diagonal matrix whose blocks each satisfy the
identity satisfies it as a whole, so D(r)^T D(r') = D(r' - r) still holds.

Crucially, P is a FULL head_dim x head_dim matrix: it mixes ACROSS the
d | h | w | leftover partition.  That mixing is exactly the generalisation
being bought -- the axes are no longer confined to disjoint channel blocks.

  HARD DEPENDENCY: this requires the CORRECT rotary construction, i.e.
  `repeat_interleave(2, dim=-1)` as used in app/vjepa_3/models/utils/modules.py.
  `src/models/utils/modules.py:42-43` instead uses `.repeat(1, 1, 1, 2)`
  (tiling), which pairs adjacent channels with DIFFERENT frequencies -- that is
  not a rotation, D(p)^T D(p') != D(p' - p), and translation invariance fails.
  The bug is acknowledged in-source there and kept for V-JEPA 2 checkpoint
  compatibility.  Never route STRING through that version.
  Pinned by test_translation_invariance_* in tests/models/test_vjepa3_string.py.

WHY CAYLEY
----------
P = (I - S)(I + S)^-1  with  S = W - W^T antisymmetric.

An antisymmetric real S has purely imaginary eigenvalues, so -1 is never an
eigenvalue of S, so (I + S) is ALWAYS invertible and P is always orthogonal --
no constraint, no projection step, no risk of drifting off the manifold.

W = 0  =>  S = 0  =>  P = I  =>  bitwise-identical to plain RoPE.

That zero-init is the whole warm-start story: a V-JEPA 2.1 checkpoint loaded
into a use_string=True model reports `string.weight` as the only missing key,
leaves it at zero, and training starts EXACTLY at the 2.1 solution.

Note Cayley parameterises the subset of SO(D) reachable without an eigenvalue
of -1 (it cannot express a rotation by exactly pi).  This is what the paper
does; it is a deliberate limitation, not an oversight.

DTYPE
-----
`torch.linalg.solve` has no bfloat16 kernel -- bf16 inputs raise
NotImplementedError('"lu_cpu" not implemented for BFloat16'), and cuSOLVER has
the same gap.  It is also not on the autocast op list, so today fp32 inputs
inside autocast(bfloat16) return fp32 anyway.  We nonetheless force fp32 AND
wrap the solve in an explicit autocast(enabled=False) guard, because relying on
an op's absence from the autocast list is a time-bomb.  Measured on torch 2.13:
orthogonality error 7.2e-7 fp32.  Note (P @ P.T) IS autocast-listed, so any
orthogonality assertion must be evaluated OUTSIDE autocast or it degrades to
~1.3e-3 and looks like a bug.
"""

import torch
import torch.nn as nn


class CayleySTRING(nn.Module):
    """Learned orthogonal pre-rotation shared across attention heads.

    Args:
        head_dim: width of one attention head (64 for ViT-L/g and vit_tiny).
        num_layers: number of independent bases.  Pass `depth` for a distinct
            P per transformer block, or 1 to share one P across the whole
            stack (`string_share_across_layers: true`).
        init_std: std of the normal init on W.  0.0 (default) => P = I =>
            exact V-JEPA 2.1 behaviour at step 0.

    Shape:
        weight: [num_layers, head_dim, head_dim]
        forward() -> [num_layers, head_dim, head_dim], float32

    SHARED ACROSS HEADS by design: the STRING paper reports that sharing these
    parameters across attention heads, rather than learning them separately,
    gave slight gains in BOTH efficiency and quality.  Cost here is
    head_dim**2 = 4096 params per layer (98K for a 24-layer ViT-L, ~0.03% of
    the model).

    OWNED AT MODEL LEVEL, not by each Block.  VisionTransformer holds ONE of
    these and calls it once per forward, then hands P[i] to block i.  That
    placement is deliberate:
      * one batched [L,D,D] solve instead of L separate 64x64 solves;
      * the solve sits OUTSIDE the activation-checkpoint boundary, so it is
        not recomputed during backward;
      * a single greppable parameter name for the weight-decay exclusion in
        app/vjepa_3/utils.py:init_opt;
      * it is used unconditionally, which keeps the used-parameter set
        constant -- see the DDP note below.

    DDP / static_graph
    ------------------
    The encoder is wrapped DistributedDataParallel(encoder, static_graph=True)
    (app/vjepa_3/train.py).  MEASURED on torch 2.13, the two failure modes are
    NOT the same, and the distinction matters:

      (A) a parameter consistently unused on a rank, identical every iteration
          -> ACCEPTED.  static_graph supports unused parameters provided the
             used-set is constant.  But it is not harmless: that rank still
             joins the allreduce contributing nothing, so the averaged
             gradient is scaled down by the fraction of ranks that skipped it.
             A silent change in effective learning rate, not a crash.

      (B) usage varying ACROSS ITERATIONS on one rank
          -> RuntimeError("...your training graph has changed in this
             iteration... not compatible with static_graph set to True") on
             that rank, and a hang on its peer.

    So the rule is "the used-parameter set must be CONSTANT ACROSS ITERATIONS
    on each rank", not the stronger "must be used on every step".  Note
    V-JEPA's `img_rank_ratio` split (train.py:263-289) pins each rank to one
    modality for the whole run, so a modality-gated P would land in case (A)
    and degrade quietly rather than fail loudly -- which is worse.

    Computing P unconditionally avoids both.  Do NOT gate it on T, on mode, on
    modality, or on n_registers.  And never reach for the `y + 0.0 * P.sum()`
    trick to silence a static_graph error: it restores a static graph by
    defeating the very check that was protecting you.

    Both cases are pinned, with a negative control, by
    tests/distributed/test_vjepa3_ddp_static_graph.py.
    """

    def __init__(self, head_dim: int, num_layers: int = 1, init_std: float = 0.0):
        super().__init__()
        if head_dim <= 0 or num_layers <= 0:
            raise ValueError(
                f"head_dim and num_layers must be positive, got "
                f"head_dim={head_dim}, num_layers={num_layers}"
            )
        self.head_dim = head_dim
        self.num_layers = num_layers
        # zero-init => S = 0 => P = I => exact V-JEPA 2.1 baseline at step 0
        self.weight = nn.Parameter(torch.zeros(num_layers, head_dim, head_dim))
        if init_std > 0.0:
            nn.init.normal_(self.weight, std=init_std)

    def forward(self) -> torch.Tensor:
        """Materialise the orthogonal bases. Returns [num_layers, D, D] fp32."""
        # .float() is explicit so this survives a future model.bfloat16():
        # linalg.solve has no bf16 kernel and would raise, not silently degrade.
        w = self.weight.float()
        s = w - w.transpose(-1, -2)  # antisymmetric
        eye = torch.eye(self.head_dim, device=w.device, dtype=w.dtype).expand_as(s)
        # Defensive guard -- see the DTYPE note in the module docstring.
        with torch.autocast(device_type=w.device.type, enabled=False):
            return torch.linalg.solve(eye + s, eye - s)

    def basis_for_layer(self, bases: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Select block `layer_idx`'s basis, honouring layer sharing.

        With num_layers == 1 every layer gets the same P (the modulo collapses
        to 0), so callers need no special case for the shared configuration.
        """
        return bases[layer_idx % bases.shape[0]]

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, num_layers={self.num_layers}"


def apply_string(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Left-multiply each token's feature vector by the orthogonal basis P.

    Args:
        x: [B, num_heads, N, head_dim] -- q or k, ANY dtype.
        basis: [head_dim, head_dim] orthogonal P, or None for a no-op.

    Returns: same shape and dtype as x.

    `x` stores each token as a ROW vector in its last dim, so the column-vector
    product `P q` is written `x @ P^T`.

    Applied to ALL N tokens, including a leading cls token and trailing
    registers, which `rotate_queries_or_keys` deliberately leaves unrotated.
    That is correct and intentional: P is a change of basis, not a positional
    signal.  For any pair of tokens that both skip rotation the score becomes
    q^T P^T P k = q^T k -- provably unchanged from baseline.  Slicing P around
    those tokens instead would break the basis symmetry and make cls-vs-patch
    scores inconsistent.  Pinned by
    test_cls_and_register_scores_unchanged_by_string.
    """
    if basis is None:
        return x
    return torch.matmul(x, basis.to(x.dtype).transpose(-1, -2))
