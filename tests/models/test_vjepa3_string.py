# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Stage-3/4 gates: Cayley-STRING.

See app/vjepa_3/models/utils/string.py for the derivation.  The claims these
tests pin, in order of how badly a regression would hurt:

  1. use_string=False is bitwise V-JEPA 2.1 and adds no state_dict keys.
  2. Zero-init gives P = I exactly, so a 2.1 checkpoint warm-starts at the
     2.1 solution.
  3. P is orthogonal (this is what makes the outer P^T cancel).
  4. Attention scores are TRANSLATION INVARIANT with STRING on.  This is the
     load-bearing property; it fails if anyone swaps in the tiled RoPE from
     src/models/utils/modules.py.
  5. Gradients w.r.t. W are antisymmetric-effective -- a structural check that
     the Cayley parameterisation is wired correctly.

NOTE ON MODEL WIDTH: vit_tiny is embed_dim=192 / num_heads=3 => head_dim=64,
the same head_dim as ViT-L and ViT-g, so d_dim=h_dim=w_dim=20 with 4 unrotated
dims.  The tiny model reproduces production rotary geometry exactly.
"""

import unittest

import torch

import app.vjepa_2_1.models.vision_transformer as vit_2_1
import app.vjepa_3.models.vision_transformer as vit_3
from app.vjepa_3.models.utils.modules import RoPEAttention
from app.vjepa_3.models.utils.string import CayleySTRING, apply_string

HEAD_DIM = 64
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


class TestCayleySTRINGModule(unittest.TestCase):
    def test_zero_init_gives_identity(self):
        """W = 0 => S = 0 => P = I.  The whole warm-start story."""
        bases = CayleySTRING(HEAD_DIM, num_layers=4)()
        eye = torch.eye(HEAD_DIM).expand_as(bases)
        torch.testing.assert_close(bases, eye, rtol=0, atol=0)

    def test_P_is_orthogonal(self):
        """Orthogonality is what makes the outer P^T cancel in the score."""
        for num_layers in (1, 12):
            with self.subTest(num_layers=num_layers):
                m = CayleySTRING(HEAD_DIM, num_layers, init_std=0.1)
                bases = m()
                eye = torch.eye(HEAD_DIM).expand_as(bases)
                torch.testing.assert_close(bases @ bases.mT, eye, atol=1e-5, rtol=1e-4)
                dets = torch.linalg.det(bases)
                torch.testing.assert_close(dets, torch.ones_like(dets), atol=1e-4, rtol=1e-4)

    def test_orthogonality_must_be_measured_outside_autocast(self):
        """Pinned gotcha: P @ P.T IS autocast-listed; the solve is not.

        Inside autocast(bfloat16) the orthogonality residual degrades by ~3
        orders of magnitude and looks like a broken parameterisation.  It is
        not -- it is the ASSERTION that got downcast.  Anyone debugging an
        orthogonality failure should check this first.
        """
        m = CayleySTRING(HEAD_DIM, 1, init_std=0.1)
        bases = m()
        eye = torch.eye(HEAD_DIM).expand_as(bases)
        err_outside = (bases @ bases.mT - eye).abs().max().item()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            err_inside = (bases @ bases.mT - eye).abs().max().item()
        self.assertLess(err_outside, 1e-5)
        self.assertGreater(err_inside, 10 * err_outside)

    def test_solve_stays_fp32_under_bf16_autocast(self):
        """linalg.solve has no bf16 kernel; bf16 inputs raise, not degrade."""
        m = CayleySTRING(HEAD_DIM, 1, init_std=0.1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            bases = m()
        self.assertIs(bases.dtype, torch.float32)

        eye = torch.eye(HEAD_DIM)
        with self.assertRaises(NotImplementedError):
            torch.linalg.solve(eye.bfloat16(), eye.bfloat16())

    def test_basis_for_layer_handles_sharing(self):
        shared = CayleySTRING(HEAD_DIM, num_layers=1, init_std=0.1)
        bases = shared()
        for i in range(5):
            torch.testing.assert_close(
                shared.basis_for_layer(bases, i), bases[0], rtol=0, atol=0
            )

        per_layer = CayleySTRING(HEAD_DIM, num_layers=3, init_std=0.1)
        bases = per_layer()
        for i in range(3):
            torch.testing.assert_close(
                per_layer.basis_for_layer(bases, i), bases[i], rtol=0, atol=0
            )
        self.assertFalse(torch.allclose(bases[0], bases[1]))

    def test_apply_string_none_is_identity(self):
        x = torch.randn(2, 3, 7, HEAD_DIM)
        torch.testing.assert_close(apply_string(x, None), x, rtol=0, atol=0)

    def test_apply_string_preserves_inner_products(self):
        """P orthogonal => <Pq, Pk> == <q, k>.  Basis change, not distortion."""
        bases = CayleySTRING(HEAD_DIM, 1, init_std=0.1)()[0]
        q = torch.randn(2, 3, 7, HEAD_DIM)
        k = torch.randn(2, 3, 7, HEAD_DIM)
        before = q @ k.transpose(-1, -2)
        after = apply_string(q, bases) @ apply_string(k, bases).transpose(-1, -2)
        torch.testing.assert_close(before, after, atol=1e-4, rtol=1e-4)

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            CayleySTRING(0, 1)
        with self.assertRaises(ValueError):
            CayleySTRING(HEAD_DIM, 0)


class TestTranslationInvariance(unittest.TestCase):
    """The load-bearing property: scores depend only on r' - r.

    <P^T D(r) P q, P^T D(r') P k> = q^T P^T D(r'-r) P k

    Fails if the tiled RoPE from src/models/utils/modules.py is ever used --
    tiling pairs adjacent channels with different frequencies, so D is not a
    rotation and D(p)^T D(p') != D(p'-p).
    """

    GRID_T, GRID_H, GRID_W = 3, 5, 5

    def _flat_ids(self, t0, h0, w0, extent=(2, 3, 3)):
        """Flat token ids for a sub-block anchored at (t0, h0, w0)."""
        et, eh, ew = extent
        return torch.tensor(
            [
                (t0 + t) * self.GRID_H * self.GRID_W + (h0 + h) * self.GRID_W + (w0 + w)
                for t in range(et)
                for h in range(eh)
                for w in range(ew)
            ],
            dtype=torch.long,
        ).unsqueeze(0)

    def _scores(self, attn, x, ids):
        """Pre-softmax attention scores for tokens at the given grid ids."""
        with torch.no_grad():
            _, attn_mat = attn(
                x,
                mask=ids,
                H_patches=self.GRID_H,
                W_patches=self.GRID_W,
                return_attn=True,
                string_P=self.basis,
            )
        return attn_mat

    def _run(self, use_string):
        torch.manual_seed(0)
        attn = RoPEAttention(
            dim=192,
            num_heads=3,
            grid_size=self.GRID_H,
            use_sdpa=False,  # need the explicit attention matrix
        ).eval()
        self.basis = (
            CayleySTRING(HEAD_DIM, 1, init_std=0.1)()[0] if use_string else None
        )

        torch.manual_seed(1)
        n_tokens = 2 * 3 * 3
        x = torch.randn(1, n_tokens, 192)

        base = self._scores(attn, x, self._flat_ids(0, 0, 0))
        for shift in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)):
            with self.subTest(use_string=use_string, shift=shift):
                shifted = self._scores(attn, x, self._flat_ids(*shift))
                torch.testing.assert_close(shifted, base, atol=1e-4, rtol=1e-3)

    def test_invariant_without_string(self):
        self._run(use_string=False)

    def test_invariant_with_string(self):
        self._run(use_string=True)


class TestSTRINGInViT(unittest.TestCase):
    def _model(self, tree=vit_3, **overrides):
        torch.manual_seed(0)
        return tree.vit_tiny(**{**BASE_KWARGS, **overrides}).eval()

    def test_use_string_false_is_bitwise_2_1(self):
        """Default off => byte-identical to V-JEPA 2.1, no new keys."""
        m_2_1 = self._model(tree=vit_2_1)
        m_3 = self._model(use_string=False)
        self.assertEqual(
            sorted(m_2_1.state_dict().keys()), sorted(m_3.state_dict().keys())
        )
        m_3.load_state_dict(m_2_1.state_dict())
        x = _video()
        with torch.no_grad():
            for a, b in zip(m_2_1(x), m_3(x)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_zero_init_string_is_numerically_baseline(self):
        """P = I at step 0 => warm start from a 2.1 checkpoint is exact.

        Not bitwise: the extra matmul by the identity reorders floating-point
        operations.  Numerically indistinguishable is the correct bar.
        """
        m_2_1 = self._model(tree=vit_2_1)
        m_3 = self._model(use_string=True, string_init_std=0.0)
        torch.testing.assert_close(
            m_3.string.weight, torch.zeros_like(m_3.string.weight), rtol=0, atol=0
        )

        missing, unexpected = m_3.load_state_dict(m_2_1.state_dict(), strict=False)
        self.assertEqual(list(unexpected), [])
        self.assertEqual(
            [k for k in missing if not k.startswith("string.")],
            [],
            "warm start should be missing ONLY string.* keys",
        )

        x = _video()
        with torch.no_grad():
            for a, b in zip(m_2_1(x), m_3(x)):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_trained_string_changes_the_output(self):
        base = self._model(use_string=False)
        m = self._model(use_string=True, string_init_std=0.1)
        m.load_state_dict(base.state_dict(), strict=False)
        x = _video()
        with torch.no_grad():
            self.assertFalse(torch.allclose(base(x)[-1], m(x)[-1]))

    def test_layer_sharing_shapes(self):
        per_layer = self._model(use_string=True)
        self.assertEqual(per_layer.string.weight.shape, (len(per_layer.blocks), HEAD_DIM, HEAD_DIM))
        shared = self._model(use_string=True, string_share_across_layers=True)
        self.assertEqual(shared.string.weight.shape, (1, HEAD_DIM, HEAD_DIM))

    def test_string_is_used_on_both_image_and_video(self):
        """DDP static_graph requires the parameter be used on EVERY step.

        V-JEPA mixes T=1 image ranks and video ranks in one allreduce group,
        so a P that only participated on video would deadlock.  Assert a
        non-zero gradient reaches string.weight for both input shapes.
        """
        for shape in ((1, 3, 1, CROP, CROP), (1, 3, NUM_FRAMES, CROP, CROP)):
            with self.subTest(T=shape[2]):
                m = self._model(use_string=True, string_init_std=0.1)
                m.train()
                m(torch.randn(*shape))[-1].sum().backward()
                grad = m.string.weight.grad
                self.assertIsNotNone(grad)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().max().item(), 0.0)

    def test_gradient_is_antisymmetric_effective(self):
        """S = W - W^T ignores W's symmetric part, so dL/dW must be antisymmetric.

        A structural check on the parameterisation: if this fails, the Cayley
        transform is not wired the way the derivation assumes.
        """
        m = self._model(use_string=True, string_init_std=0.1)
        m.train()
        m(_video())[-1].sum().backward()
        g = m.string.weight.grad
        torch.testing.assert_close(g, -g.mT, atol=1e-6, rtol=1e-5)

    def test_activation_checkpointing_matches_plain(self):
        """The solve sits outside the checkpoint boundary; results must agree."""
        plain = self._model(use_string=True, string_init_std=0.1)
        ckpt = self._model(
            use_string=True, string_init_std=0.1, use_activation_checkpointing=True
        )
        ckpt.load_state_dict(plain.state_dict())
        x = _video()
        with torch.no_grad():
            torch.testing.assert_close(plain(x)[-1], ckpt(x)[-1], atol=1e-6, rtol=1e-6)


class TestSTRINGInPredictor(unittest.TestCase):
    """The predictor is flagged separately (use_string_predictor).

    Its RoPE never receives T/H_patches/W_patches (predictor.py forward) and it
    runs on re-sorted token sequences, so it is a distinct positional regime --
    and a distinct wiring path that needs its own coverage.
    """

    def _predictor(self, **overrides):
        from app.vjepa_3.models.predictor import vit_predictor

        torch.manual_seed(0)
        return vit_predictor(
            img_size=(CROP, CROP), patch_size=16, num_frames=NUM_FRAMES,
            tubelet_size=2, embed_dim=192, predictor_embed_dim=96, depth=4,
            num_heads=3, use_rope=True, use_mask_tokens=True, num_mask_tokens=2,
            **overrides,
        )

    def _run(self, pred):
        # predictor_embed consumes the concatenated deep-supervision levels
        x = torch.randn(1, 40, pred.predictor_embed[0].in_features)
        out, _ = pred(x, torch.arange(0, 40).unsqueeze(0), torch.arange(40, 64).unsqueeze(0))
        return out

    def test_predictor_string_receives_gradient(self):
        pred = self._predictor(use_string=True, string_init_std=0.1)
        # head_dim here is 96/3 = 32, not 64 -- the predictor is narrower
        self.assertEqual(pred.string.head_dim, 32)
        self._run(pred).sum().backward()
        grad = pred.string.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().max().item(), 0.0)
        torch.testing.assert_close(grad, -grad.mT, atol=1e-6, rtol=1e-5)

    def test_predictor_default_adds_no_keys(self):
        self.assertIsNone(self._predictor().string)
        self.assertEqual(
            sorted(self._predictor().state_dict().keys()),
            sorted(
                k
                for k in self._predictor(use_string=True).state_dict()
                if not k.startswith("string.")
            ),
        )

    def test_unsupported_pred_depth_raises_clearly(self):
        """V-JEPA 2.1 raised a bare UnboundLocalError naming the wrong thing.

        `all_hierarchical_layers` was only bound for depth in
        {4, 8, 12, 20, 24, 40}; any other value fell through with no else and
        failed on the next line with a message about an unbound local rather
        than about an unsupported pred_depth.
        """
        with self.assertRaises(ValueError) as ctx:
            self._predictor_depth(7)
        self.assertIn("pred_depth", str(ctx.exception).lower().replace("predictor depth", "pred_depth"))

    def _predictor_depth(self, depth):
        from app.vjepa_3.models.predictor import vit_predictor

        return vit_predictor(
            img_size=(CROP, CROP), patch_size=16, num_frames=NUM_FRAMES,
            tubelet_size=2, embed_dim=192, predictor_embed_dim=96,
            depth=depth, num_heads=3, use_rope=True,
        )


class TestInitOptWeightDecay(unittest.TestCase):
    """Stage-4 gate: STRING params must be excluded from weight decay."""

    def _towers(self, use_string):
        torch.manual_seed(0)
        enc = vit_3.vit_tiny(**BASE_KWARGS, use_string=use_string, string_init_std=0.1)
        pred = vit_3.vit_tiny(**BASE_KWARGS, use_string=use_string, string_init_std=0.1)
        return enc, pred

    def _init_opt(self, enc, pred):
        from app.vjepa_3.utils import init_opt

        return init_opt(
            is_anneal=False,
            encoder=enc,
            predictor=pred,
            iterations_per_epoch=10,
            start_lr=1e-4,
            ref_lr=1e-3,
            warmup=1,
            num_epochs=2,
            wd=0.04,
            final_wd=0.04,
        )

    def test_string_params_are_wd_excluded(self):
        enc, pred = self._towers(use_string=True)
        optimizer = self._init_opt(enc, pred)[0]

        string_ids = {id(p) for n, p in enc.named_parameters() if "string.weight" in n}
        string_ids |= {id(p) for n, p in pred.named_parameters() if "string.weight" in n}
        self.assertEqual(len(string_ids), 2)

        for group in optimizer.param_groups:
            ids = {id(p) for p in group["params"]}
            if ids & string_ids:
                self.assertTrue(ids <= string_ids, "STRING params share a group")
                self.assertTrue(group.get("WD_exclude"))
                self.assertEqual(group["weight_decay"], 0)
                string_ids -= ids
        self.assertEqual(string_ids, set(), "some STRING params were not grouped")

    def test_wd_schedule_never_decays_string(self):
        enc, pred = self._towers(use_string=True)
        optimizer, _, _, wd_scheduler = self._init_opt(enc, pred)
        string_groups = [
            g for g in optimizer.param_groups
            if g.get("WD_exclude") and g["weight_decay"] == 0 and len(g["params"]) == 1
        ]
        self.assertGreaterEqual(len(string_groups), 2)
        for _ in range(100):
            wd_scheduler.step()
        for g in string_groups:
            self.assertEqual(g["weight_decay"], 0)
        decay_group = optimizer.param_groups[0]
        self.assertNotEqual(decay_group["weight_decay"], 0)

    def test_groups_partition_when_string_disabled(self):
        """The partition assertion must also hold on the baseline path."""
        enc, pred = self._towers(use_string=False)
        optimizer = self._init_opt(enc, pred)[0]
        grouped = sum(len(g["params"]) for g in optimizer.param_groups)
        total = len(list(enc.parameters())) + len(list(pred.parameters()))
        self.assertEqual(grouped, total)


if __name__ == "__main__":
    unittest.main()
