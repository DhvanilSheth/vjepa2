# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Stage-1/2 gates: RoPE frequency control in V-JEPA 3.

`build_omega` factors out the rotary frequency table that V-JEPA 2.1 computed
inline inside `rotate_queries_or_keys`, so the base ("max wavelength") can be
configured per axis.  The contract is that the default reproduces 2.1 exactly.

See tests/models/test_vjepa3_parity.py for why vit_tiny's head_dim=64 is the
production geometry.
"""

import unittest

import torch

from app.vjepa_3.models.utils.modules import (
    DEFAULT_ROPE_THETA,
    build_omega,
    rotate_queries_or_keys,
)

# RoPEAttention splits head_dim=64 into d_dim=h_dim=w_dim=20.
AXIS_DIM = 20


def _legacy_omega(dim, dtype):
    """The exact expression V-JEPA 2.1 inlines in rotate_queries_or_keys."""
    omega = torch.arange(dim // 2, dtype=dtype)
    omega /= dim / 2.0
    return 1.0 / 10000**omega


class TestBuildOmega(unittest.TestCase):
    def test_matches_legacy_bitwise(self):
        """Default theta must reproduce 2.1's inline table, bit for bit."""
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for dim in (AXIS_DIM, 64, 128):
                with self.subTest(dtype=dtype, dim=dim):
                    torch.testing.assert_close(
                        build_omega(dim, DEFAULT_ROPE_THETA, dtype=dtype),
                        _legacy_omega(dim, dtype),
                        rtol=0,
                        atol=0,
                    )

    def test_shape_and_range(self):
        om = build_omega(AXIS_DIM)
        self.assertEqual(om.shape, (AXIS_DIM // 2,))
        self.assertAlmostEqual(om[0].item(), 1.0, places=6)
        # strictly decreasing
        self.assertTrue(bool((om[1:] < om[:-1]).all()))

    def test_lower_theta_widens_the_live_band(self):
        """The motivating claim: theta=10000 leaves most channels dead.

        A channel is 'live' over a grid of P positions if it sweeps more than
        0.2 rad across the whole extent.  At 8 temporal positions (16 frames /
        tubelet 2), of 10 frequency channels:

            theta=10000 ->  4 live   (V-JEPA 2.1 default)
            theta= 1000 ->  6 live
            theta=  100 ->  8 live   (STRING paper / RoPE-Mixed init)

        Note even theta=100 does not make all 10 live -- lowering the base
        widens the usable band but does not eliminate the tail.
        """
        max_pos = 7  # 8 temporal positions at 16 frames / tubelet 2

        def live(theta):
            return int(((build_omega(AXIS_DIM, theta) * max_pos) > 0.2).sum())

        self.assertEqual(live(DEFAULT_ROPE_THETA), 4)
        self.assertEqual(live(1000.0), 6)
        self.assertEqual(live(100.0), 8)


class TestRotateQueriesOrKeys(unittest.TestCase):
    """The omega= argument must be a pure injection point at Stage 1."""

    def _inputs(self, dtype=torch.float32):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 16, AXIS_DIM, dtype=dtype)
        pos = torch.arange(16, dtype=dtype).expand(2, 3, 16)
        return x, pos

    def test_explicit_default_omega_equals_none(self):
        """Passing the default table explicitly == letting it default."""
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x, pos = self._inputs(dtype)
                out_none = rotate_queries_or_keys(x, pos, 0, False, omega=None)
                out_expl = rotate_queries_or_keys(
                    x, pos, 0, False, omega=build_omega(AXIS_DIM, dtype=dtype)
                )
                torch.testing.assert_close(out_none, out_expl, rtol=0, atol=0)

    def test_fp32_precompute_does_NOT_match_legacy_bf16(self):
        """Pinned negative result -- this is why the default path stays omega=None.

        Building the table in fp32 and casting to bf16 is NOT bitwise equal to
        building it in bf16 throughout: `omega /= n` and `1/theta**omega` round
        differently at the two precisions.  At AXIS_DIM=20 they diverge in 4 of
        10 channels, e.g. index 4: 0.025146484375 (fp32->bf16) vs
        0.0250244140625 (native bf16).

        CONSEQUENCE FOR RoPEAttention: a precomputed fp32 omega buffer would
        silently break bitwise parity with V-JEPA 2.1 under autocast, even at
        theta=10000.  So RoPEAttention holds `None` whenever an axis uses the
        default base, and only materialises a table when the config actually
        deviates.  See TestRoPEAttentionThetaWiring.
        """
        x, pos = self._inputs(torch.bfloat16)
        out_none = rotate_queries_or_keys(x, pos, 0, False, omega=None)
        out_fp32 = rotate_queries_or_keys(
            x, pos, 0, False, omega=build_omega(AXIS_DIM, dtype=torch.float32)
        )
        self.assertFalse(torch.equal(out_none, out_fp32))
        # ...but they are numerically close -- this is rounding, not a bug
        torch.testing.assert_close(out_none, out_fp32, rtol=2e-2, atol=2e-2)

    def test_different_theta_changes_output(self):
        x, pos = self._inputs()
        base = rotate_queries_or_keys(x, pos, 0, False, omega=None)
        lowered = rotate_queries_or_keys(
            x, pos, 0, False, omega=build_omega(AXIS_DIM, 100.0)
        )
        self.assertFalse(torch.allclose(base, lowered))

    def test_registers_and_cls_are_never_rotated(self):
        """cls (leading) and registers (trailing) pass through untouched."""
        x, pos_full = self._inputs()
        n_reg = 2
        # pos only covers the context tokens
        pos = pos_full[..., : 16 - n_reg - 1]
        out = rotate_queries_or_keys(x, pos, n_registers=n_reg, has_cls_first=True)
        torch.testing.assert_close(out[..., :1, :], x[..., :1, :], rtol=0, atol=0)
        torch.testing.assert_close(out[..., -n_reg:, :], x[..., -n_reg:, :], rtol=0, atol=0)


class TestRoPEAttentionThetaWiring(unittest.TestCase):
    """Stage-2 gate: rope_theta / rope_temporal_scale reach RoPEAttention."""

    @staticmethod
    def _attn(**kwargs):
        from app.vjepa_3.models.utils.modules import RoPEAttention

        torch.manual_seed(0)
        return RoPEAttention(dim=192, num_heads=3, grid_size=4, **kwargs).eval()

    def test_default_theta_leaves_omega_none(self):
        """The parity-preserving design: default base => no precomputed table.

        See test_fp32_precompute_does_NOT_match_legacy_bf16 for why holding a
        buffer here would break bitwise parity with V-JEPA 2.1.
        """
        attn = self._attn()
        self.assertIsNone(attn.omega_d)
        self.assertIsNone(attn.omega_h)
        self.assertIsNone(attn.omega_w)

    def test_omega_buffers_are_non_persistent(self):
        """Buffers must stay out of the state_dict so strict loads still work."""
        attn = self._attn(rope_theta=100.0)
        self.assertIsNotNone(attn.omega_d)
        keys = attn.state_dict().keys()
        for name in ("omega_d", "omega_h", "omega_w"):
            self.assertNotIn(name, keys)

    def test_per_axis_theta(self):
        """[100, 10000, 10000] rebases temporal only -- an ablation arm."""
        attn = self._attn(rope_theta=[100.0, 10000.0, 10000.0])
        self.assertIsNotNone(attn.omega_d)   # temporal deviates -> materialised
        self.assertIsNone(attn.omega_h)      # spatial at default -> legacy path
        self.assertIsNone(attn.omega_w)
        torch.testing.assert_close(
            attn.omega_d, build_omega(attn.d_dim, 100.0), rtol=0, atol=0
        )

    def test_anisotropic_theta_raises(self):
        """theta_h != theta_w would give the model a preferred image axis."""
        with self.assertRaises(ValueError):
            self._attn(rope_theta=[100.0, 100.0, 200.0])
        with self.assertRaises(ValueError):
            self._attn(rope_theta=[1.0, 2.0, 3.0])

    def test_malformed_theta_raises(self):
        with self.assertRaises(ValueError):
            self._attn(rope_theta=[100.0, 100.0])      # wrong length
        with self.assertRaises(ValueError):
            self._attn(rope_theta=0.5)                 # base must exceed 1

    def test_temporal_scale_stored_and_defaults_to_noop(self):
        self.assertEqual(self._attn().rope_temporal_scale, 1.0)
        self.assertEqual(self._attn(rope_temporal_scale=0.25).rope_temporal_scale, 0.25)


class TestEndToEndThetaWiring(unittest.TestCase):
    """The full yaml -> ViT hop chain, which is where keys silently vanish."""

    NUM_FRAMES, CROP = 8, 64

    def _model(self, **overrides):
        import app.vjepa_3.models.vision_transformer as vit_3

        torch.manual_seed(0)
        return vit_3.vit_tiny(
            img_size=(self.CROP, self.CROP),
            num_frames=self.NUM_FRAMES,
            use_rope=True,
            **overrides,
        ).eval()

    def test_theta_reaches_every_block(self):
        model = self._model(rope_theta=100.0, rope_temporal_scale=0.5)
        for blk in model.blocks:
            self.assertEqual(blk.attn.rope_theta, (100.0, 100.0, 100.0))
            self.assertEqual(blk.attn.rope_temporal_scale, 0.5)
            self.assertIsNotNone(blk.attn.omega_d)

    def test_defaults_change_nothing(self):
        model = self._model()
        for blk in model.blocks:
            self.assertEqual(blk.attn.rope_theta, (DEFAULT_ROPE_THETA,) * 3)
            self.assertEqual(blk.attn.rope_temporal_scale, 1.0)
            self.assertIsNone(blk.attn.omega_d)

    def test_theta_and_temporal_scale_change_the_output(self):
        """A knob that reaches the model must actually move the numbers."""
        x = torch.randn(1, 3, self.NUM_FRAMES, self.CROP, self.CROP)
        base = self._model()
        with torch.no_grad():
            out_base = base(x)[-1]

        for overrides in ({"rope_theta": 100.0}, {"rope_temporal_scale": 0.25}):
            with self.subTest(**overrides):
                model = self._model(**overrides)
                model.load_state_dict(base.state_dict())
                with torch.no_grad():
                    out = model(x)[-1]
                self.assertFalse(
                    torch.allclose(out_base, out),
                    f"{overrides} did not change the output -- key not threaded",
                )

    def test_state_dict_unchanged_by_theta(self):
        """New knobs add no parameters, so checkpoints stay compatible."""
        self.assertEqual(
            sorted(self._model().state_dict().keys()),
            sorted(self._model(rope_theta=100.0).state_dict().keys()),
        )


if __name__ == "__main__":
    unittest.main()
