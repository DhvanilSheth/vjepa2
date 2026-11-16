# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Stage-6 gates: `norm_type` / `norm_out_type` (models/utils/norms.py).

The claims these tests pin:

  1. DyT and Derf compute the published formulas with the published inits,
     and those inits survive the ViT's `self.apply(self._init_weights)`.
  2. `norm_type` swaps EVERY block pre-norm; `norm_out_type` swaps the output
     norms and follows `norm_type` unless set.  Nothing is left half-converted.
  3. `norm_type: layernorm` (and the default None) is the 2.1 model, bitwise.
  4. The new scalars (alpha, shift) are never weight-decayed, without any
     DyT-specific code in init_opt.
  5. Under bf16 autocast the squashing runs in fp32, like LayerNorm does.
"""

import copy
import math
import unittest

import torch
import torch.nn as nn

import app.vjepa_3.models.vision_transformer as vit_3
from app.vjepa_3.models.utils.norms import (
    DynamicErf,
    DynamicTanh,
    build_norm_layer,
)

NUM_FRAMES, CROP = 8, 64
DEPTH = 12  # vit_tiny
PRED_DEPTH = 4
BASE_KWARGS = dict(
    img_size=(CROP, CROP),
    num_frames=NUM_FRAMES,
    use_rope=True,
    img_temporal_dim_size=1,
)


def _video(seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, 3, NUM_FRAMES, CROP, CROP)


def _model(**overrides):
    torch.manual_seed(0)
    return vit_3.vit_tiny(**BASE_KWARGS, **overrides).eval()


def _predictor(**overrides):
    from app.vjepa_3.models.predictor import vit_predictor

    torch.manual_seed(0)
    return vit_predictor(
        img_size=(CROP, CROP), patch_size=16, num_frames=NUM_FRAMES,
        tubelet_size=2, embed_dim=192, predictor_embed_dim=96, depth=PRED_DEPTH,
        num_heads=3, use_rope=True, use_mask_tokens=True, num_mask_tokens=2,
        **overrides,
    )


def _count(model, cls):
    return sum(isinstance(m, cls) for m in model.modules())


class TestNormModules(unittest.TestCase):
    def test_dyt_closed_form(self):
        torch.manual_seed(0)
        m = DynamicTanh(8)
        with torch.no_grad():
            m.alpha.fill_(1.7)
            m.weight.normal_()
            m.bias.normal_()
        x = torch.randn(3, 5, 8)
        torch.testing.assert_close(m(x), torch.tanh(1.7 * x) * m.weight + m.bias)

    def test_derf_closed_form(self):
        torch.manual_seed(0)
        m = DynamicErf(8)
        with torch.no_grad():
            m.alpha.fill_(0.9)
            m.shift.fill_(-0.3)
            m.weight.normal_()
            m.bias.normal_()
        x = torch.randn(3, 5, 8)
        expected = torch.erf(0.9 * x - 0.3) * m.weight + m.bias
        torch.testing.assert_close(m(x), expected)
        # erf, not tanh: spot-check one scalar against math.erf
        one = DynamicErf(1)
        self.assertAlmostEqual(one(torch.tensor([[1.0]])).item(), math.erf(0.5), places=6)

    def test_published_inits(self):
        for cls in (DynamicTanh, DynamicErf):
            m = cls(16)
            self.assertEqual(m.alpha.shape, (1,))
            self.assertEqual(m.alpha.item(), 0.5)
            self.assertTrue((m.weight == 1).all() and (m.bias == 0).all())
        self.assertEqual(DynamicErf(16).shift.item(), 0.0)

    def test_outputs_are_bounded_by_the_affine(self):
        """The defining property vs LayerNorm: a hard bound, no statistics."""
        x = torch.randn(4, 7, 16) * 1e4
        for cls in (DynamicTanh, DynamicErf):
            self.assertLessEqual(cls(16)(x).abs().max().item(), 1.0)

    def test_build_norm_layer(self):
        ln = build_norm_layer("layernorm")(16)
        self.assertIsInstance(ln, nn.LayerNorm)
        self.assertEqual(ln.eps, 1e-6)  # what every ViT factory passes
        self.assertIsInstance(build_norm_layer("dyt")(16), DynamicTanh)
        self.assertIsInstance(build_norm_layer("derf")(16), DynamicErf)
        with self.assertRaisesRegex(ValueError, "Unknown norm_type"):
            build_norm_layer("rmsnorm")

    def test_runs_in_fp32_under_bf16_autocast(self):
        x = torch.randn(2, 5, 16).to(torch.bfloat16)
        for cls in (DynamicTanh, DynamicErf):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = cls(16)(x)
            self.assertEqual(out.dtype, torch.float32)
            self.assertTrue(torch.isfinite(out).all())


class TestNormInTheEncoder(unittest.TestCase):
    def test_layernorm_and_default_are_bitwise_identical(self):
        base, explicit = _model(), _model(norm_type="layernorm")
        self.assertEqual(sorted(base.state_dict()), sorted(explicit.state_dict()))
        x = _video()
        with torch.no_grad():
            torch.testing.assert_close(base(x), explicit(x), rtol=0, atol=0)

    def test_norm_type_replaces_every_norm(self):
        for name, cls in (("dyt", DynamicTanh), ("derf", DynamicErf)):
            with self.subTest(norm_type=name):
                model = _model(norm_type=name)
                self.assertEqual(_count(model, nn.LayerNorm), 0)
                # two pre-norms per block + four deep-supervision output norms
                self.assertEqual(_count(model, cls), 2 * DEPTH + 4)

    def test_norm_out_type_is_independent(self):
        model = _model(norm_type="dyt", norm_out_type="layernorm")
        self.assertEqual(_count(model, DynamicTanh), 2 * DEPTH)
        self.assertEqual(_count(model, nn.LayerNorm), 4)
        for norm in model.norms_block:
            self.assertIsInstance(norm, nn.LayerNorm)

        model = _model(norm_out_type="derf")
        self.assertEqual(_count(model, nn.LayerNorm), 2 * DEPTH)
        for norm in model.norms_block:
            self.assertIsInstance(norm, DynamicErf)

    def test_inits_survive_init_weights(self):
        """self.apply(_init_weights) must not clobber alpha = 0.5."""
        for name in ("dyt", "derf"):
            model = _model(norm_type=name)
            for mod in model.modules():
                if isinstance(mod, (DynamicTanh, DynamicErf)):
                    self.assertEqual(mod.alpha.item(), 0.5)
                    self.assertTrue((mod.weight == 1).all() and (mod.bias == 0).all())

    def test_changes_the_output_and_trains(self):
        x = _video()
        with torch.no_grad():
            out_ln = _model()(x)
        for name in ("dyt", "derf"):
            with self.subTest(norm_type=name):
                model = _model(norm_type=name).train()
                out = model(x, training=True)
                self.assertTrue(torch.isfinite(out).all())
                out.sum().backward()
                for mod in model.modules():
                    if isinstance(mod, (DynamicTanh, DynamicErf)):
                        self.assertIsNotNone(mod.alpha.grad)
                        self.assertTrue(torch.isfinite(mod.alpha.grad).all())
                with torch.no_grad():
                    self.assertFalse(torch.allclose(out_ln, model.eval()(x)))

    def test_dyt_and_derf_are_different_functions(self):
        x = _video()
        dyt, derf = _model(norm_type="dyt"), _model(norm_type="derf")
        with torch.no_grad():
            self.assertFalse(torch.allclose(dyt(x), derf(x)))

    def test_full_forward_under_bf16_autocast(self):
        x = _video()
        for name in ("dyt", "derf"):
            with self.subTest(norm_type=name):
                model = _model(norm_type=name)
                with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                    out = model(x)
                self.assertTrue(torch.isfinite(out.float()).all())

    def test_unknown_norm_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown norm_type"):
            _model(norm_type="batchnorm")
        with self.assertRaisesRegex(ValueError, "Unknown norm_type"):
            _model(norm_out_type="batchnorm")

    def test_ema_copy_carries_the_new_parameters(self):
        """train.py builds the target encoder with copy.deepcopy(encoder)."""
        enc = _model(norm_type="derf")
        target = copy.deepcopy(enc)
        self.assertEqual(len(list(enc.parameters())), len(list(target.parameters())))
        self.assertIn("blocks.0.norm1.shift", dict(target.named_parameters()))


class TestNormInThePredictor(unittest.TestCase):
    def test_predictor_norms_follow_the_keys(self):
        pred = _predictor(norm_type="dyt")
        self.assertEqual(_count(pred, nn.LayerNorm), 0)
        self.assertEqual(_count(pred, DynamicTanh), 2 * PRED_DEPTH + 1)
        self.assertIsInstance(pred.predictor_norm, DynamicTanh)

        pred = _predictor(norm_type="dyt", norm_out_type="layernorm")
        self.assertIsInstance(pred.predictor_norm, nn.LayerNorm)
        self.assertEqual(_count(pred, DynamicTanh), 2 * PRED_DEPTH)

    def test_predictor_default_keys_unchanged(self):
        self.assertEqual(
            sorted(_predictor().state_dict()),
            sorted(_predictor(norm_type="layernorm").state_dict()),
        )


class TestNormHopChainAndOptimizer(unittest.TestCase):
    def _towers(self, **overrides):
        from app.vjepa_3.utils import init_video_model

        torch.manual_seed(0)
        return init_video_model(
            device="cpu", patch_size=16, max_num_frames=NUM_FRAMES, tubelet_size=2,
            model_name="vit_tiny", crop_size=CROP, pred_depth=PRED_DEPTH,
            pred_num_heads=3, pred_embed_dim=96, use_mask_tokens=True,
            num_mask_tokens=2, use_rope=True, img_temporal_dim_size=1,
            modality_embedding=True, **overrides,
        )

    def test_keys_reach_both_towers(self):
        enc, pred = self._towers(norm_type="derf", norm_out_type="layernorm")
        self.assertEqual(_count(enc, DynamicErf), 2 * DEPTH)
        self.assertEqual(_count(enc, nn.LayerNorm), 4)
        self.assertEqual(_count(pred, DynamicErf), 2 * PRED_DEPTH)
        self.assertIsInstance(pred.backbone.predictor_norm, nn.LayerNorm)

    def test_alpha_and_shift_are_never_weight_decayed(self):
        from app.vjepa_3.utils import init_opt

        enc, pred = self._towers(norm_type="derf")
        optimizer, _, _, wd_scheduler = init_opt(
            is_anneal=False, encoder=enc, predictor=pred, iterations_per_epoch=10,
            start_lr=1e-4, ref_lr=1e-3, warmup=1, num_epochs=2, wd=0.04, final_wd=0.04,
        )
        targets = {
            id(p)
            for model in (enc, pred)
            for n, p in model.named_parameters()
            if n.endswith((".alpha", ".shift"))
        }
        self.assertEqual(len(targets), 2 * (2 * DEPTH + 4) + 2 * (2 * PRED_DEPTH + 1))

        for _ in range(3):
            wd_scheduler.step()
        found = 0
        for group in optimizer.param_groups:
            ids = {id(p) for p in group["params"]}
            if ids & targets:
                found += len(ids & targets)
                self.assertEqual(group["weight_decay"], 0)
                self.assertTrue(group.get("WD_exclude", False))
        self.assertEqual(found, len(targets), "some alpha/shift params are in no group")


if __name__ == "__main__":
    unittest.main()
