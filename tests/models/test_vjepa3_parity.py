# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Stage-0 gate for the V-JEPA 3 fork.

`app/vjepa_3/` is a fork of `app/vjepa_2_1/`.  Every V-JEPA 3 feature is
gated behind a config flag whose default reproduces V-JEPA 2.1 exactly, so
that `app/vjepa_2_1/` can serve as an untouched A/B control arm.

These tests are the contract that makes that claim checkable: at default
settings the two trees must produce identical parameter sets and bitwise
identical outputs.  They must keep passing after every later stage.

NOTE ON MODEL WIDTH: `vit_tiny` is embed_dim=192 / num_heads=3 => head_dim=64,
which is the SAME head_dim as ViT-L (1024/16) and ViT-g (1408/22).  RoPE
splits head_dim into d_dim=h_dim=w_dim=int(2*((64//3)//2))=20 with 4 dims
left unrotated, so the tiny model reproduces the production rotary geometry
exactly.  Do not "upgrade" these tests to a wider model -- the point is that
this width is representative, and it runs on CPU in seconds.
"""

import unittest

import torch

import app.vjepa_2_1.models.vision_transformer as vit_2_1
import app.vjepa_3.models.vision_transformer as vit_3


# Small but representative geometry: 64x64 / patch 16 -> 4x4 spatial,
# 8 frames / tubelet 2 -> 4 temporal, so 64 tokens.  Runs on CPU in ms.
NUM_FRAMES = 8
CROP = 64
# V-JEPA 2.1's ViT only supports use_rope=True; see TestVJepa2_1KnownBreakage.
BASE_KWARGS = dict(
    img_size=(CROP, CROP),
    num_frames=NUM_FRAMES,
    use_rope=True,
    img_temporal_dim_size=1,
)


def _build_pair(**overrides):
    """Build the same ViT from both trees under an identical RNG stream."""
    kwargs = {**BASE_KWARGS, **overrides}
    torch.manual_seed(0)
    model_2_1 = vit_2_1.vit_tiny(**kwargs)
    torch.manual_seed(0)
    model_3 = vit_3.vit_tiny(**kwargs)
    return model_2_1.eval(), model_3.eval()


def _assert_outputs_equal(test, out_a, out_b):
    """forward() returns a list of per-level (deep-supervision) outputs."""
    test.assertEqual(len(out_a), len(out_b))
    for a, b in zip(out_a, out_b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


class TestVJepa3ForkParity(unittest.TestCase):
    VIDEO_SHAPE = (1, 3, NUM_FRAMES, CROP, CROP)
    IMAGE_SHAPE = (1, 3, 1, CROP, CROP)

    def test_state_dict_keys_identical(self):
        """The fork must register exactly the same parameters as 2.1."""
        m_2_1, m_3 = _build_pair()
        self.assertEqual(
            sorted(m_2_1.state_dict().keys()),
            sorted(m_3.state_dict().keys()),
        )

    def test_state_dict_values_identical(self):
        """Same seed => same init, so the fork changes no initializer."""
        m_2_1, m_3 = _build_pair()
        sd_2_1, sd_3 = m_2_1.state_dict(), m_3.state_dict()
        for key in sd_2_1:
            torch.testing.assert_close(sd_2_1[key], sd_3[key], rtol=0, atol=0)

    def test_video_forward_bitwise_identical(self):
        """The headline gate: identical video outputs, bit for bit."""
        m_2_1, m_3 = _build_pair()
        # copy weights across so the comparison isolates the CODE path
        m_3.load_state_dict(m_2_1.state_dict())

        torch.manual_seed(1)
        x = torch.randn(*self.VIDEO_SHAPE)
        with torch.no_grad():
            _assert_outputs_equal(self, m_2_1(x), m_3(x))

    def test_image_forward_bitwise_identical(self):
        """T=1 image path (V-JEPA 2.1 trains on mixed image+video batches)."""
        m_2_1, m_3 = _build_pair()
        m_3.load_state_dict(m_2_1.state_dict())

        torch.manual_seed(2)
        x = torch.randn(*self.IMAGE_SHAPE)
        with torch.no_grad():
            _assert_outputs_equal(self, m_2_1(x), m_3(x))

    def test_masked_forward_bitwise_identical(self):
        """Masked path -- the one JEPA pretraining actually exercises."""
        m_2_1, m_3 = _build_pair()
        m_3.load_state_dict(m_2_1.state_dict())

        masks = [torch.tensor([3, 11, 27, 40], dtype=torch.int64).unsqueeze(0)]
        torch.manual_seed(3)
        x = torch.randn(*self.VIDEO_SHAPE)
        with torch.no_grad():
            _assert_outputs_equal(self, m_2_1(x, masks=masks), m_3(x, masks=masks))


class TestVJepa2_1KnownBreakage(unittest.TestCase):
    """Pre-existing V-JEPA 2.1 defects that the fork reproduces faithfully.

    These are pinned deliberately.  If one of them starts passing, someone has
    changed behaviour in `app/vjepa_3/` that `app/vjepa_2_1/` does not share,
    and the A/B control arm is no longer a true control.
    """

    def test_use_rope_false_is_broken_in_both_trees(self):
        """`use_rope=False` raises: `self.pos_embed` is read but never assigned.

        `app/vjepa_2_1/models/vision_transformer.py:275-276` does
            if not self.use_rope:
                pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        but unlike `src/models/vision_transformer.py:84`, the 2.1 fork dropped
        `_init_pos_embed` and never creates `self.pos_embed`.  So the additive
        sin/cos branch -- and therefore everything in
        `app/vjepa_3/models/utils/pos_embs.py` -- is unreachable dead code in
        this tree.  All shipped configs set `use_rope: true`.
        """
        kwargs = {**BASE_KWARGS, "use_rope": False}
        x = torch.randn(1, 3, NUM_FRAMES, CROP, CROP)
        for name, module in (("vjepa_2_1", vit_2_1), ("vjepa_3", vit_3)):
            with self.subTest(tree=name):
                model = module.vit_tiny(**kwargs).eval()
                with self.assertRaises(AttributeError):
                    with torch.no_grad():
                        model(x)


class TestVJepa3EmbedDimTable(unittest.TestCase):
    """V-JEPA 3 fixes the unbound-`embed_dim_encoder` NameError in 2.1.

    `app/vjepa_2_1/train.py:117-124` mapped only vit_large /
    vit_giant_xformers / vit_gigantic_xformers; its `else` branch printed a
    message and fell through, leaving `embed_dim_encoder` unbound and raising
    NameError at :591.  configs/train_2_1/vitb16/*.yaml (model_name:
    vit_base) therefore cannot train.  V-JEPA 3 replaces that chain with an
    exhaustive table plus a loud failure.
    """

    def test_table_matches_every_factory(self):
        """Every vit_* factory must appear in the table with the right width."""
        import ast
        import inspect

        source = inspect.getsource(vit_3)
        tree = ast.parse(source)

        actual = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("vit_"):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.keyword) and sub.arg == "embed_dim":
                    actual[node.name] = ast.literal_eval(sub.value)

        # mirror of EMBED_DIM_BY_MODEL_NAME in app/vjepa_3/train.py
        import re
        from pathlib import Path

        train_src = (
            Path(__file__).resolve().parents[2] / "app" / "vjepa_3" / "train.py"
        ).read_text()
        block = re.search(
            r"EMBED_DIM_BY_MODEL_NAME = \{(.*?)\n    \}", train_src, re.S
        )
        self.assertIsNotNone(block, "EMBED_DIM_BY_MODEL_NAME not found in train.py")
        table = dict(
            (m.group(1), int(m.group(2)))
            for m in re.finditer(r'"(vit_\w+)":\s*(\d+)', block.group(1))
        )

        self.assertEqual(
            table,
            actual,
            "EMBED_DIM_BY_MODEL_NAME is out of sync with the vit_* factories",
        )


if __name__ == "__main__":
    unittest.main()
