# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Stage-5 gates: the `ffn_type` flag (app/vjepa_3/models/utils/ffn.py).

The claims these tests pin:

  1. `ffn_type=None` (default) is the untouched V-JEPA 2.1 path.  The Stage-0
     parity tests already cover default-vs-2.1; here we add that an EXPLICIT
     `gelu` / `swiglu` is bitwise the same model as the legacy switch.
  2. `relu2` has exactly the parameters of `gelu` and differs only in the
     nonlinearity.
  3. The key reaches the Blocks of BOTH towers through `init_video_model`
     (the hop chain where keys silently vanish into **kwargs).
  4. Contradictory or unknown settings raise instead of silently picking one.
"""

import unittest

import torch

import app.vjepa_2_1.models.vision_transformer as vit_2_1
import app.vjepa_3.models.vision_transformer as vit_3
from app.vjepa_3.models.utils.ffn import (
    FFN_TYPES,
    GLU_GATES,
    SquaredReLU,
    resolve_ffn_type,
)
from app.vjepa_3.models.utils.modules import MLP, SwiGLUFFN, build_ffn

NUM_FRAMES, CROP = 8, 64
BASE_KWARGS = dict(
    img_size=(CROP, CROP),
    num_frames=NUM_FRAMES,
    use_rope=True,
    img_temporal_dim_size=1,
)


def _video(seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, 3, NUM_FRAMES, CROP, CROP)


def _model(tree=vit_3, **overrides):
    torch.manual_seed(0)
    return tree.vit_tiny(**BASE_KWARGS, **overrides).eval()


def _assert_same_model(test, a, b, x):
    sd_a, sd_b = a.state_dict(), b.state_dict()
    test.assertEqual(sorted(sd_a), sorted(sd_b))
    for key in sd_a:
        torch.testing.assert_close(sd_a[key], sd_b[key], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)


class TestFFNPieces(unittest.TestCase):
    def test_squared_relu_closed_form(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])
        torch.testing.assert_close(
            SquaredReLU()(x), torch.tensor([0.0, 0.0, 0.0, 0.25, 9.0])
        )

    def test_build_ffn_types(self):
        self.assertIsInstance(build_ffn("gelu", 32, 128), MLP)
        relu2 = build_ffn("relu2", 32, 128)
        self.assertIsInstance(relu2, MLP)
        self.assertIsInstance(relu2.act, SquaredReLU)
        for name, gate in GLU_GATES.items():
            with self.subTest(ffn_type=name):
                ffn = build_ffn(name, 32, 128)
                self.assertIsInstance(ffn, SwiGLUFFN)
                self.assertIs(ffn.gate_fn, gate)

    def test_glu_gate_is_actually_applied(self):
        """gate(fc1(x)) * fc2(x) -- recomputed by hand for each gate."""
        x = torch.randn(2, 5, 32)
        for name, gate in GLU_GATES.items():
            with self.subTest(ffn_type=name):
                ffn = build_ffn(name, 32, 128)
                expected = ffn.fc3(gate(ffn.fc1(x)) * ffn.fc2(x))
                torch.testing.assert_close(ffn(x), expected, rtol=0, atol=0)

    def test_resolve_rejects_unknown_and_conflicts(self):
        self.assertIsNone(resolve_ffn_type(None, use_silu=False))
        self.assertIsNone(resolve_ffn_type(None, use_silu=True))
        self.assertEqual(resolve_ffn_type("swiglu", use_silu=True), "swiglu")
        with self.assertRaisesRegex(ValueError, "Unknown ffn_type"):
            resolve_ffn_type("gelu2")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_ffn_type("relu2", use_silu=True)
        with self.assertRaisesRegex(ValueError, "Unknown ffn_type"):
            build_ffn("nope", 32, 128)


class TestFFNInTheEncoder(unittest.TestCase):
    def test_explicit_gelu_is_bitwise_default_and_2_1(self):
        x = _video()
        _assert_same_model(self, _model(), _model(ffn_type="gelu"), x)
        _assert_same_model(self, _model(vit_2_1), _model(ffn_type="gelu"), x)

    def test_explicit_swiglu_is_bitwise_use_silu_and_2_1(self):
        x = _video()
        _assert_same_model(self, _model(use_silu=True), _model(ffn_type="swiglu"), x)
        _assert_same_model(
            self, _model(vit_2_1, use_silu=True), _model(ffn_type="swiglu"), x
        )

    def test_relu2_same_parameters_different_function(self):
        base, relu2 = _model(), _model(ffn_type="relu2")
        self.assertEqual(sorted(base.state_dict()), sorted(relu2.state_dict()))
        self.assertEqual(
            sum(p.numel() for p in base.parameters()),
            sum(p.numel() for p in relu2.parameters()),
        )
        for blk in relu2.blocks:
            self.assertIsInstance(blk.mlp.act, SquaredReLU)
        # identical weights, so any output difference is the nonlinearity
        relu2.load_state_dict(base.state_dict())
        x = _video()
        with torch.no_grad():
            self.assertFalse(torch.allclose(base(x), relu2(x)))

    def test_every_ffn_type_runs_and_backpropagates(self):
        x = _video()
        for ffn_type in FFN_TYPES:
            with self.subTest(ffn_type=ffn_type):
                model = _model(ffn_type=ffn_type).train()
                out = model(x, training=True)
                self.assertTrue(torch.isfinite(out).all())
                out.sum().backward()
                grad = model.blocks[0].mlp.fc1.weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().max().item(), 0.0)

    def test_glu_gates_differ_from_each_other(self):
        """Same weights, different gate => different output (key is live)."""
        x = _video()
        swiglu = _model(ffn_type="swiglu")
        with torch.no_grad():
            out_swiglu = swiglu(x)
        for other in ("geglu", "reglu"):
            with self.subTest(ffn_type=other):
                model = _model(ffn_type=other)
                model.load_state_dict(swiglu.state_dict())
                with torch.no_grad():
                    self.assertFalse(torch.allclose(out_swiglu, model(x)))

    def test_conflict_and_unknown_raise_at_construction(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _model(use_silu=True, ffn_type="relu2")
        with self.assertRaisesRegex(ValueError, "Unknown ffn_type"):
            _model(ffn_type="squared_relu")


class TestFFNHopChain(unittest.TestCase):
    """config value -> init_video_model -> factory -> ViT -> Block."""

    def _towers(self, **overrides):
        from app.vjepa_3.utils import init_video_model

        torch.manual_seed(0)
        enc, pred = init_video_model(
            device="cpu",
            patch_size=16,
            max_num_frames=NUM_FRAMES,
            tubelet_size=2,
            model_name="vit_tiny",
            crop_size=CROP,
            pred_depth=4,
            pred_num_heads=3,
            pred_embed_dim=96,
            use_mask_tokens=True,
            num_mask_tokens=2,
            use_rope=True,
            img_temporal_dim_size=1,
            modality_embedding=True,
            **overrides,
        )
        return enc.backbone, pred.backbone

    def test_default_is_gelu_mlp_in_both_towers(self):
        enc, pred = self._towers()
        for blk in list(enc.blocks) + list(pred.predictor_blocks):
            self.assertIsInstance(blk.mlp, MLP)
            self.assertIsInstance(blk.mlp.act, torch.nn.GELU)

    def test_encoder_and_predictor_keys_are_independent(self):
        enc, pred = self._towers(ffn_type="relu2")
        self.assertIsInstance(enc.blocks[0].mlp.act, SquaredReLU)
        self.assertIsInstance(pred.predictor_blocks[0].mlp.act, torch.nn.GELU)

        enc, pred = self._towers(pred_ffn_type="swiglu")
        self.assertIsInstance(enc.blocks[0].mlp, MLP)
        for blk in pred.predictor_blocks:
            self.assertIsInstance(blk.mlp, SwiGLUFFN)


if __name__ == "__main__":
    unittest.main()
