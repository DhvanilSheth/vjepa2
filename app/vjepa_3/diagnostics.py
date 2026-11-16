# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""V-JEPA 3: measurement-only diagnostics.  Nothing here feeds the loss.

SIGReg / Epps-Pulley isotropy statistic
---------------------------------------
LeJEPA (Balestriero & LeCun, arXiv 2511.08544) argues the best embedding
distribution for downstream probes is an isotropic Gaussian, and measures the
distance to it with SIGReg: project the embeddings onto random unit
directions, and for each 1-D projection compare its empirical characteristic
function with that of N(0, 1) (the Epps-Pulley test).

LeJEPA uses this as a training loss.  HERE IT IS ONLY A READING: V-JEPA 3
still trains with the EMA target and the L1 prediction loss.  We log the
statistic so ablation arms can be compared on how isotropic their features
are, and so drift toward collapse shows up before a probe is run.

HOW TO READ THE NUMBER
----------------------
For samples that really are N(0, I), the statistic does not go to zero: its
expectation is a constant, `gaussian_floor()` (about 1.06 with the default
integration grid), whatever the sample size.  For anything else it grows
LINEARLY with the number of samples N.  So:

  * compare readings only at the same N (`isotropy_report` always subsamples
    to exactly `max_tokens` and reports the N it used);
  * a value near the floor means "indistinguishable from isotropic Gaussian
    at this N"; a full collapse to a point gives a value of order N.

`sigreg_raw` is the statistic on the features as they are (sensitive to mean,
scale and shape).  `sigreg_std` first z-scores every dimension over the
sample, so it ignores per-dimension mean and scale and responds only to shape
and to correlation between dimensions.  Raw high + std low means "right shape,
wrong scale"; both high means the shape itself is off.

The two readings have DIFFERENT floors.  Z-scoring fits the mean from the same
sample that is then tested, which removes part of the sampling noise, so a
perfect Gaussian scores lower on `sigreg_std` than on `sigreg_raw`:
`gaussian_floor(standardized=True)` (about 0.58) versus `gaussian_floor()`
(about 1.06).  The 0.58 figure is for wide features (D in the hundreds, as in
any ViT); for very narrow features the scale is effectively fitted too and
the floor drops further, toward 0.34 at D = 1.

CAVEATS
-------
* The test assumes independent samples.  Patch tokens from one clip are not
  independent, so treat this as a relative reading across arms and across
  time, not a calibrated hypothesis test.
* Random projections of high-dimensional data look Gaussian easily (central
  limit theorem), so the statistic is blind to non-Gaussianity that lives in
  single coordinates.  It is sensitive to wrong scale, wrong mean, low rank
  and heavy tails -- the failure modes that matter for collapse.
"""

import torch


def _grid(t_max, n_points, device):
    t = torch.linspace(-t_max, t_max, n_points, device=device)
    return t, torch.exp(-0.5 * t**2)


def gaussian_floor(standardized=False, t_max=5.0, n_points=17):
    """Expected statistic for exact N(0, I) samples, for any sample size.

    E[N * |ecf(t) - phi(t)|^2] = 1 - phi(t)^2 for the complex empirical
    characteristic function, so the floor is the weighted integral of that.

    standardized=True gives the floor of `sigreg_std`: fitting the mean from
    the sample removes the t^2 * phi(t)^2 component of that variance.  See
    the module docstring for when this applies.
    """
    t, phi = _grid(t_max, n_points, device="cpu")
    variance = 1.0 - phi**2
    if standardized:
        variance = variance - t**2 * phi**2
    return float(torch.trapz(variance * phi, t))


@torch.no_grad()
def epps_pulley_statistic(x, num_slices=256, seed=0, t_max=5.0, n_points=17):
    """Sliced Epps-Pulley statistic of `x` against N(0, I).

    LeJEPA Algorithm 1 for a single process (no all_reduce), averaged over
    slices.  `x` is [N, D].  Returns a 0-dim float32 tensor.

    `seed` fixes the projection directions.  LeJEPA resamples them every
    step because it is optimising; for a reading we keep them FIXED so two
    measurements differ only because the features do.
    """
    if x.ndim != 2:
        raise ValueError(f"expected [N, D], got shape {tuple(x.shape)}")
    # matmul is autocast-listed: without this the projections would be
    # computed in bf16 inside train.py's autocast block.
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.detach().float()
        n = x.size(0)
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        directions = torch.randn((x.size(1), num_slices), generator=gen, device=x.device)
        directions /= directions.norm(p=2, dim=0)
        t, phi = _grid(t_max, n_points, x.device)
        x_t = (x @ directions).unsqueeze(2) * t  # [N, slices, T]
        ecf = torch.complex(x_t.cos(), x_t.sin()).mean(0)  # [slices, T]
        err = (ecf - phi).abs().square() * phi
        return (torch.trapz(err, t, dim=1) * n).mean()


@torch.no_grad()
def isotropy_report(feats, max_tokens=4096, seed=0, num_slices=256):
    """Isotropy reading for a feature tensor of shape [..., D].

    Flattens to [N, D], subsamples to at most `max_tokens` rows (`seed` picks
    the rows; projection directions are always the same), and returns
    {"sigreg_raw", "sigreg_std", "n"} as Python numbers.
    """
    x = feats.detach().reshape(-1, feats.shape[-1]).float()
    if x.size(0) > max_tokens:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        x = x[torch.randperm(x.size(0), generator=gen, device=x.device)[:max_tokens]]
    standardized = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
    return {
        "sigreg_raw": float(epps_pulley_statistic(x, num_slices=num_slices)),
        "sigreg_std": float(epps_pulley_statistic(standardized, num_slices=num_slices)),
        "n": int(x.size(0)),
    }
