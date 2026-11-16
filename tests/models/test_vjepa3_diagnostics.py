# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Stage-7 gates: the SIGReg / Epps-Pulley isotropy reading.

A diagnostic is only worth logging if it moves when the thing it measures
moves, so most of this file is negative controls: distributions that are NOT
an isotropic Gaussian must score far above the Gaussian floor.
"""

import unittest

import torch

from app.vjepa_3.diagnostics import (
    epps_pulley_statistic,
    gaussian_floor,
    isotropy_report,
)

N, D = 4096, 64


def _gaussian(seed=0, n=N, d=D):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _algorithm_1(x, global_step, num_slices=256):
    """LeJEPA Algorithm 1, transcribed, with all_reduce = identity."""
    g = torch.Generator()
    g.manual_seed(global_step)
    A = torch.randn((x.size(1), num_slices), generator=g)
    A /= A.norm(p=2, dim=0)
    t = torch.linspace(-5, 5, 17)
    exp_f = torch.exp(-0.5 * t**2)
    x_t = (x @ A).unsqueeze(2) * t
    ecf = (1j * x_t).exp().mean(0)
    err = (ecf - exp_f).abs().square().mul(exp_f)
    return torch.trapz(err, t, dim=1) * x.size(0)


class TestEppsPulley(unittest.TestCase):
    def test_matches_lejepa_algorithm_1(self):
        x = _gaussian() * 1.3 + 0.2
        torch.testing.assert_close(
            epps_pulley_statistic(x, seed=7), _algorithm_1(x, 7).mean(), rtol=1e-5, atol=1e-6
        )

    def test_gaussian_sits_at_the_floor_for_any_n(self):
        floor = gaussian_floor()
        self.assertAlmostEqual(floor, 1.06, delta=0.02)
        for n in (512, 4096):
            with self.subTest(n=n):
                stat = float(epps_pulley_statistic(_gaussian(n=n)))
                self.assertAlmostEqual(stat, floor, delta=0.15 * floor)

    def test_non_gaussian_inputs_score_far_above_the_floor(self):
        floor = gaussian_floor()
        g = _gaussian()
        radial = torch.rand(N, 1, generator=torch.Generator().manual_seed(1)) * 3
        low_rank = g.clone()
        low_rank[:, 4:] = 0
        cases = {
            "collapsed": torch.zeros(N, D),
            "wrong_scale": g * 2.0,
            "wrong_mean": g + 1.0,
            "low_rank": low_rank,
            "heavy_tailed": g * radial,
        }
        for name, x in cases.items():
            with self.subTest(case=name):
                stat = float(epps_pulley_statistic(x))
                self.assertGreater(stat, 20 * floor, f"{name}: {stat:.2f}")

    def test_grows_linearly_with_n_when_not_gaussian(self):
        small = float(epps_pulley_statistic(_gaussian(n=1024) * 2.0))
        large = float(epps_pulley_statistic(_gaussian(n=4096) * 2.0))
        self.assertAlmostEqual(large / small, 4.0, delta=0.6)

    def test_fixed_directions_are_deterministic(self):
        x = _gaussian() * 1.5
        self.assertEqual(
            float(epps_pulley_statistic(x, seed=3)), float(epps_pulley_statistic(x, seed=3))
        )
        self.assertNotEqual(
            float(epps_pulley_statistic(x, seed=3)), float(epps_pulley_statistic(x, seed=4))
        )

    def test_immune_to_bf16_autocast(self):
        """train.py calls this inside autocast(bfloat16)."""
        x = _gaussian() * 1.5
        expected = float(epps_pulley_statistic(x))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            inside = float(epps_pulley_statistic(x.to(torch.bfloat16)))
        # the INPUT was rounded to bf16, so allow that much and no more
        self.assertAlmostEqual(inside, expected, delta=0.02 * expected)

    def test_rejects_wrong_rank(self):
        with self.assertRaisesRegex(ValueError, r"expected \[N, D\]"):
            epps_pulley_statistic(torch.randn(2, 3, 4))


class TestIsotropyReport(unittest.TestCase):
    def test_flattens_and_subsamples_to_a_fixed_n(self):
        feats = torch.randn(8, 1024, D, generator=torch.Generator().manual_seed(0))
        report = isotropy_report(feats, max_tokens=4096)
        self.assertEqual(report["n"], 4096)
        self.assertEqual(set(report), {"sigreg_raw", "sigreg_std", "n"})
        self.assertEqual(isotropy_report(feats[:1, :100])["n"], 100)

    def test_std_variant_separates_scale_from_shape(self):
        floor = gaussian_floor()
        # right shape, wrong scale: raw is far off, standardised is at the floor
        scaled = isotropy_report(_gaussian() * 3.0 + 2.0)
        self.assertGreater(scaled["sigreg_raw"], 20 * floor)
        # ...at ITS floor, which is lower: the mean was fitted from the sample
        std_floor = gaussian_floor(standardized=True)
        self.assertAlmostEqual(std_floor, 0.58, delta=0.02)
        self.assertAlmostEqual(scaled["sigreg_std"], std_floor, delta=0.15 * std_floor)
        # wrong shape (strongly correlated dimensions): both are far off
        shared = _gaussian(n=N, d=1).expand(N, D)
        correlated = isotropy_report(shared + 0.1 * _gaussian(seed=2))
        self.assertGreater(correlated["sigreg_std"], 20 * floor)

    def test_subsample_seed_changes_rows_not_directions(self):
        feats = _gaussian(n=3 * N) * 1.5
        a = isotropy_report(feats, seed=0)
        b = isotropy_report(feats, seed=1)
        self.assertNotEqual(a["sigreg_raw"], b["sigreg_raw"])
        self.assertAlmostEqual(a["sigreg_raw"], b["sigreg_raw"], delta=0.25 * a["sigreg_raw"])


if __name__ == "__main__":
    unittest.main()
