# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""V-JEPA 3: optional distribution regularisers on the encoder output.

OFF BY DEFAULT.  With `loss.reg_type: null` nothing in this file is called and
training is V-JEPA 2.1's, bit for bit (tests/train/test_vjepa3_train_loop.py).

WHAT THIS ADDS
--------------
V-JEPA has no loss term that forbids collapse; it relies on the EMA teacher,
the stop-gradient and LayerNorm.  LeJEPA (Balestriero & LeCun, arXiv
2511.08544) and VISReg (Wu, Balestriero & Levine, arXiv 2606.02572) instead
add an explicit term that pulls the embedding distribution toward an
isotropic Gaussian, using many random 1-D projections ("sketching"):

  sigreg  LeJEPA's SIGReg: for each projection, the distance between the
          empirical characteristic function and that of N(0, 1) (Epps-Pulley).
  visreg  VISReg: centre + scale + shape.  Shape is the sliced 2-Wasserstein
          distance to N(0, 1) of the per-dimension-standardised features;
          scale is (1 - std)^2 per dimension.  The paper's argument for it is
          that SIGReg's gradient fades as the features collapse, which is
          when a correction is needed most.

HOW IT IS USED HERE -- AND HOW THAT DIFFERS FROM THE PAPERS
-----------------------------------------------------------
    loss = L_pred + lambda * L_ctx + reg_weight * L_reg

1. ADDED TO the EMA objective, not replacing it.  Both papers remove the
   teacher.  We keep it: our H-JEPA notes (V3_STRING/README.md section 9.7)
   record that a marginal regulariser alone lets a video model keep the
   background and drop the moving object.
2. Applied to PATCH TOKENS of the context encoder's last output level.  Both
   papers regularise one pooled vector per image, after a projector MLP, with
   batches of 128-512 images.  A batch here is 16 clips -- far too few
   vectors to shape a distribution -- so we use the tokens (thousands per
   batch).  Tokens from one clip are NOT independent, and neither paper
   tested that case.  This is the open question the arm exists to answer.
3. No projector.  The constraint therefore acts on the backbone features
   themselves, which is stronger than in either paper.
4. Statistics are per process.  On more than one GPU each rank regularises
   its own tokens (LeJEPA all-reduces across ranks); the ladder is single-GPU.
5. `sigreg_loss` is LeJEPA's Algorithm 1 WITHOUT the final "* N".  That factor
   makes the value grow with the number of samples, so a weight tuned at one
   token count would mean something else at another.  Without it the value
   is bounded whatever the input (below 7.6): about 0 for a Gaussian, 0.41
   when every token is the same point at the origin, 0.82 for centred
   features spread far too wide, more for a large common offset.
   LeJEPA's lambda at N samples corresponds to reg_weight = lambda * N here.

The readings in diag_r*.csv (app/vjepa_3/diagnostics.py) are measured on the
TARGET encoder with fixed projections; this loss acts on the CONTEXT encoder
with projections resampled every step.  They are related but not the same
number.
"""

import math

import torch

REG_TYPES = ("sigreg", "visreg")


def _directions(dim, num_slices, seed, device):
    """`num_slices` random unit vectors in R^dim, as the columns of [dim, K].

    Drawn on the CPU so that a seed gives the same directions on every
    device, then moved.  Resampled every step (the caller passes the global
    step as the seed), as both papers do: a fixed set would only constrain
    those particular projections.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    directions = torch.randn((dim, num_slices), generator=gen)
    directions = directions / directions.norm(p=2, dim=0)
    return directions.to(device)


def _check(x):
    if x.ndim != 2:
        raise ValueError(f"expected [N, D], got shape {tuple(x.shape)}")
    if x.size(0) < 2:
        raise ValueError(f"need at least 2 samples to regularise, got {x.size(0)}")


def sigreg_loss(x, num_slices=1024, seed=0, t_max=5.0, n_points=17):
    """Sliced Epps-Pulley distance of `x` ([N, D]) to N(0, I).  Differentiable.

    Equal to `diagnostics.epps_pulley_statistic(x) / N` for the same
    directions.  See note 5 in the module docstring for the missing "* N".
    """
    _check(x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        directions = _directions(x.size(1), num_slices, seed, x.device)
        t = torch.linspace(-t_max, t_max, n_points, device=x.device)
        phi = torch.exp(-0.5 * t**2)
        x_t = (x @ directions).unsqueeze(2) * t  # [N, K, T]
        real = x_t.cos().mean(0) - phi  # [K, T]
        imag = x_t.sin().mean(0)
        err = (real.square() + imag.square()) * phi
        return torch.trapz(err, t, dim=1).mean()


def visreg_loss(
    x,
    num_slices=1024,
    seed=0,
    scale_weight=1.0,
    shape_weight=1.0,
    center_weight=1.0,
    eps=1e-6,
    return_parts=False,
):
    """VISReg's decoupled regulariser (the paper's Algorithm 1) on `x` ([N, D]).

    Differs from the listing in two guards only: `eps` in the division (the
    paper's Eq. 2 has it, the listing drops it), and 1e-8 under the square
    root, without which the gradient is NaN when a dimension is exactly
    constant -- the collapsed case this loss is supposed to repair.
    """
    _check(x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        n = x.size(0)
        mean = x.mean(dim=0)
        center = mean.square().mean()

        centered = x - mean
        std = torch.sqrt(centered.square().mean(dim=0) + 1e-8)
        scale = (1.0 - std).square().mean()

        # Shape: standardise with a stop-gradient on the scale, so the shape
        # term cannot change the variance, then compare each sorted 1-D
        # projection with the Gaussian quantiles.
        normed = centered / (std.detach() + eps)
        directions = _directions(x.size(1), num_slices, seed, x.device)
        projected = torch.sort(normed @ directions, dim=0).values  # [N, K]
        u = torch.arange(1, n + 1, device=x.device, dtype=torch.float32) / (n + 1)
        quantiles = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
        shape = (projected - quantiles.unsqueeze(1)).square().mean()

        total = scale_weight * scale + shape_weight * shape + center_weight * center
        if return_parts:
            return total, {"scale": scale, "shape": shape, "center": center}
        return total


def build_regularizer(reg_type, num_slices=1024):
    """`reg_type` -> callable `(x, seed) -> scalar loss`, or None when off."""
    if reg_type is None:
        return None
    if reg_type == "sigreg":
        return lambda x, seed: sigreg_loss(x, num_slices=num_slices, seed=seed)
    if reg_type == "visreg":
        return lambda x, seed: visreg_loss(x, num_slices=num_slices, seed=seed)
    raise ValueError(f"Unknown reg_type {reg_type!r}. Known: {list(REG_TYPES)} or null.")


def resolve_regularizer(reg_type, reg_weight, num_slices=1024):
    """Validate the two config keys together and build the regulariser.

    A type without a weight (or the reverse) would silently do nothing, which
    in this codebase is how a run ends up labelled as something it is not.
    """
    reg_weight = float(reg_weight or 0.0)
    if reg_weight < 0:
        raise ValueError(f"loss.reg_weight must be >= 0, got {reg_weight}")
    if reg_type is None and reg_weight > 0:
        raise ValueError(
            f"loss.reg_weight is {reg_weight} but loss.reg_type is null: "
            f"set reg_type to one of {list(REG_TYPES)} or reg_weight to 0."
        )
    if reg_type is not None and reg_weight == 0:
        raise ValueError(
            f"loss.reg_type is {reg_type!r} but loss.reg_weight is 0: the "
            "regulariser would never reach the loss. Set a weight, or set "
            "reg_type to null."
        )
    return build_regularizer(reg_type, num_slices), reg_weight


def gather_reg_tokens(z, embed_dim, max_tokens=4096, seed=0):
    """Encoder outputs -> the [N, D] tokens the regulariser sees.

    `z` is what `MultiSeqWrapper` returns in training mode: a list over clip
    lengths of a list over masks of [B, n_visible, levels * D] tensors.  The
    LAST level is taken -- the final-layer features after the output norm,
    which is what the predictor consumes and what the frozen probes score.

    At most `max_tokens` rows are kept, picked uniformly over the whole batch
    (`seed` picks the rows), which bounds the cost and spreads the sample
    across clips.
    """
    tokens = torch.cat(
        [zij[:, :, -embed_dim:].reshape(-1, embed_dim) for zi in z for zij in zi]
    )
    if max_tokens and tokens.size(0) > max_tokens:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        keep = torch.randperm(tokens.size(0), generator=gen)[:max_tokens]
        tokens = tokens[keep.to(tokens.device)]
    return tokens
