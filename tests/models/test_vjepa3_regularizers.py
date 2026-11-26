# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""app/vjepa_3/regularizers.py: SIGReg and VISReg as loss terms.

Each property below has its failing case next to it (a collapsed, scaled or
shifted input), because a regulariser that returns a small number for
everything would pass a "Gaussian scores low" test on its own.
"""

import math
import unittest

import torch

from app.vjepa_3.diagnostics import epps_pulley_statistic
from app.vjepa_3.regularizers import (
    REG_TYPES,
    build_regularizer,
    gather_reg_tokens,
    resolve_regularizer,
    sigreg_loss,
    visreg_loss,
)


def _gaussian(n=2048, d=32, seed=0):
    return torch.randn(n, d, generator=torch.Generator().manual_seed(seed))


def _visreg_listing(z, directions):
    """Algorithm 1 of the VISReg paper, transcribed literally (no guards)."""
    n = z.size(0)
    mu = z.mean(dim=0)
    l_center = mu.pow(2).mean()
    z_cent = z - mu
    std = z_cent.std(dim=0, unbiased=False)
    l_scale = (1.0 - std).pow(2).mean()
    z_norm = z_cent / std.detach()
    p_sorted = torch.sort(z_norm @ directions, dim=0).values
    u = torch.arange(1, n + 1) / (n + 1)
    target = torch.distributions.Normal(0.0, 1.0).icdf(u)
    l_shape = (p_sorted - target.unsqueeze(1)).pow(2).mean()
    return l_scale + l_shape + l_center


class TestSigreg(unittest.TestCase):
    def test_is_the_diagnostic_statistic_without_the_sample_count(self):
        """Same directions, same integral: loss * N == the logged reading."""
        for n in (64, 500):
            x = _gaussian(n=n, d=24, seed=n) * 1.7 + 0.3
            loss = sigreg_loss(x, num_slices=128, seed=5)
            reading = epps_pulley_statistic(x, num_slices=128, seed=5)
            torch.testing.assert_close(loss * n, reading, rtol=1e-5, atol=1e-6)

    def test_gaussian_is_near_zero_and_failures_are_not(self):
        base = float(sigreg_loss(_gaussian(), num_slices=256))
        self.assertLess(base, 0.005)
        x = _gaussian()
        failures = {
            "collapsed to a point": torch.zeros_like(x),
            "scaled x4": x * 4.0,
            "shifted by 2": x + 2.0,
            "rank 1": x[:, :1].repeat(1, x.size(1)),
        }
        for name, bad in failures.items():
            with self.subTest(case=name):
                self.assertGreater(float(sigreg_loss(bad, num_slices=256)), 20 * base)

    def test_value_does_not_grow_with_the_sample_count(self):
        """The reason the "* N" of LeJEPA's listing is dropped (module note 5)."""
        small = float(sigreg_loss(torch.zeros(256, 16), num_slices=64))
        large = float(sigreg_loss(torch.zeros(4096, 16), num_slices=64))
        self.assertAlmostEqual(small, large, places=5)
        # closed form for a point mass: sqrt(2 pi) * (1 - 2/sqrt(2) + 1/sqrt(3))
        self.assertAlmostEqual(small, 0.409, places=2)
        # the other documented reference points
        wide = float(sigreg_loss(_gaussian(n=4096, d=16) * 1e4, num_slices=64))
        self.assertAlmostEqual(wide, 0.82, places=1)
        offset = float(sigreg_loss(torch.full((256, 16), 50.0), num_slices=64))
        self.assertGreater(offset, wide)
        self.assertLess(offset, 7.6)

    def test_gradient_pulls_toward_the_gaussian(self):
        x = (_gaussian(n=512, d=16) * 3.0).requires_grad_(True)
        before = sigreg_loss(x, num_slices=128, seed=1)
        (grad,) = torch.autograd.grad(before, x)
        self.assertTrue(torch.isfinite(grad).all())
        after = sigreg_loss(x.detach() - 50.0 * grad, num_slices=128, seed=1)
        self.assertLess(float(after), float(before))


class TestVisreg(unittest.TestCase):
    def test_matches_the_paper_listing(self):
        from app.vjepa_3.regularizers import _directions

        x = _gaussian(n=300, d=20, seed=3) * 0.6 + 0.2
        expected = _visreg_listing(x, _directions(20, 64, seed=9, device=x.device))
        torch.testing.assert_close(
            visreg_loss(x, num_slices=64, seed=9), expected, rtol=1e-4, atol=1e-6
        )

    def test_parts_respond_to_their_own_failure(self):
        x = _gaussian()
        _, good = visreg_loss(x, num_slices=128, return_parts=True)
        _, scaled = visreg_loss(x * 3.0, num_slices=128, return_parts=True)
        _, shifted = visreg_loss(x + 2.0, num_slices=128, return_parts=True)
        _, heavy = visreg_loss(x.sign() * x.abs().pow(3.0), num_slices=128, return_parts=True)
        for part in ("scale", "shape", "center"):
            self.assertLess(float(good[part]), 0.02, part)
        # scale: only the scale term moves (the shape term is standardised first)
        self.assertGreater(float(scaled["scale"]), 3.0)
        self.assertLess(float(scaled["shape"]), 0.02)
        # centre: only the centre term moves
        self.assertGreater(float(shifted["center"]), 3.0)
        self.assertLess(float(shifted["shape"]), 0.02)
        # shape: heavy tails are caught although mean and variance are standardised away
        self.assertGreater(float(heavy["shape"]), 10 * float(good["shape"]))

    def test_shape_term_has_a_stop_gradient_on_the_scale(self):
        """Eq. 2 of the paper divides by sg(std).  Without the stop-gradient the
        shape term would be a scale-invariant function of the centred
        features, and by Euler's theorem its gradient would have NO component
        along them.  With it, that component is there."""
        x = _gaussian(n=512, d=16, seed=2)
        x = (x.sign() * x.abs().pow(2.0)).requires_grad_(True)  # heavy tails
        loss = visreg_loss(x, num_slices=128, scale_weight=0.0, center_weight=0.0)
        (grad,) = torch.autograd.grad(loss, x)
        centred = (x - x.mean(0)).detach()
        radial = float((grad * centred).sum())
        self.assertGreater(abs(radial), 1e-2)

    def test_gradient_is_finite_and_alive_under_collapse(self):
        """The paper's reason for VISReg.  Exactly constant features are where
        the listing's sqrt has an infinite derivative; the guard keeps it finite.
        """
        collapsed = torch.full((256, 16), 0.5, requires_grad=True)
        (grad,) = torch.autograd.grad(visreg_loss(collapsed, num_slices=64), collapsed)
        self.assertTrue(torch.isfinite(grad).all())

        near = (_gaussian(n=256, d=16) * 1e-3).requires_grad_(True)
        (g_vis,) = torch.autograd.grad(visreg_loss(near, num_slices=64), near)
        (g_sig,) = torch.autograd.grad(sigreg_loss(near, num_slices=64), near)
        self.assertTrue(torch.isfinite(g_vis).all())
        self.assertGreater(float(g_vis.norm()), 10 * float(g_sig.norm()))

    def test_gradient_pulls_toward_the_gaussian(self):
        x = (_gaussian(n=512, d=16) * 0.2 + 1.0).requires_grad_(True)
        before = visreg_loss(x, num_slices=128, seed=1)
        (grad,) = torch.autograd.grad(before, x)
        after = visreg_loss(x.detach() - 20.0 * grad, num_slices=128, seed=1)
        self.assertLess(float(after), float(before))


class TestShared(unittest.TestCase):
    def test_seed_fixes_the_projections(self):
        x = _gaussian(n=256, d=16) * 2.0
        for name in REG_TYPES:
            reg = build_regularizer(name, num_slices=32)
            with self.subTest(reg=name):
                self.assertEqual(float(reg(x, 3)), float(reg(x, 3)))
                self.assertNotEqual(float(reg(x, 3)), float(reg(x, 4)))

    def test_half_precision_input_is_computed_in_float32(self):
        x = (_gaussian(n=128, d=16) * 2.0).half()
        for name in REG_TYPES:
            with self.subTest(reg=name):
                out = build_regularizer(name, num_slices=32)(x, 0)
                self.assertEqual(out.dtype, torch.float32)
                self.assertTrue(math.isfinite(float(out)))

    def test_bad_input_raises(self):
        for name in REG_TYPES:
            reg = build_regularizer(name)
            with self.subTest(reg=name):
                with self.assertRaises(ValueError):
                    reg(torch.zeros(4, 5, 6), 0)
                with self.assertRaises(ValueError):
                    reg(torch.zeros(1, 6), 0)

    def test_config_pairs_that_would_silently_do_nothing_raise(self):
        self.assertEqual(resolve_regularizer(None, 0.0), (None, 0.0))
        self.assertEqual(resolve_regularizer(None, None), (None, 0.0))
        reg, weight = resolve_regularizer("visreg", 0.25)
        self.assertEqual(weight, 0.25)
        self.assertTrue(callable(reg))
        with self.assertRaisesRegex(ValueError, "would never reach the loss"):
            resolve_regularizer("sigreg", 0.0)
        with self.assertRaisesRegex(ValueError, "reg_type is null"):
            resolve_regularizer(None, 0.1)
        with self.assertRaisesRegex(ValueError, "Unknown reg_type"):
            resolve_regularizer("vicreg", 0.1)
        with self.assertRaises(ValueError):
            resolve_regularizer("sigreg", -1.0)


class TestGatherTokens(unittest.TestCase):
    def test_takes_the_last_level_of_every_mask(self):
        dim = 4
        # two clip lengths x two masks; 3 levels concatenated on the last axis
        z = [
            [torch.arange(2 * 5 * 12.0).reshape(2, 5, 12), torch.ones(2, 3, 12)],
            [torch.zeros(1, 7, 12), torch.full((1, 2, 12), 9.0)],
        ]
        tokens = gather_reg_tokens(z, dim, max_tokens=0)
        self.assertEqual(tuple(tokens.shape), (2 * 5 + 2 * 3 + 7 + 2, dim))
        torch.testing.assert_close(tokens[:10], z[0][0][:, :, -dim:].reshape(-1, dim))
        self.assertTrue((tokens[-2:] == 9.0).all())

    def test_subsample_is_bounded_seeded_and_keeps_the_graph(self):
        z = [[torch.randn(4, 50, 8, requires_grad=True)]]
        a = gather_reg_tokens(z, 4, max_tokens=32, seed=1)
        b = gather_reg_tokens(z, 4, max_tokens=32, seed=1)
        c = gather_reg_tokens(z, 4, max_tokens=32, seed=2)
        self.assertEqual(tuple(a.shape), (32, 4))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        a.sum().backward()
        self.assertEqual(int((z[0][0].grad != 0).any(dim=-1).sum()), 32)
        self.assertTrue((z[0][0].grad[:, :, :4] == 0).all(), "an earlier level got gradient")


if __name__ == "__main__":
    unittest.main()
