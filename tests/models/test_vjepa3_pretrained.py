# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""app/vjepa_3/pretrained.py: loading a released V-JEPA 2.1 encoder.

No released file is opened here (none is in the repo).  What IS checked is
everything that can be checked without one: that the architecture this fork
builds for a released model is the one Meta's own loader builds for the same
file, and that the loading code is strict about keys.
"""

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from app.vjepa_3.pretrained import (
    RELEASED_2_1,
    clean_encoder_state_dict,
    load_pretrained_encoder,
    read_encoder_state_dict,
    released_2_1_pretrain_config,
)
from evals.feature_clustering_frozen.eval import build_encoder
from src.hub import backbones

REPO_ROOT = Path(__file__).resolve().parents[2]
LADDER = REPO_ROOT / "configs" / "train_3" / "ladder"


def _shapes(module):
    return {k: tuple(v.shape) for k, v in module.state_dict().items()}


class TestReleasedArchitecture(unittest.TestCase):
    def test_table_agrees_with_metas_loader(self):
        """File name -> factory, against ARCH_NAME_MAP in src/hub/backbones.py."""
        hub = {
            file_name: arch
            for name, (arch, file_name) in backbones.ARCH_NAME_MAP.items()
            if name.startswith("vjepa2_1_")
        }
        self.assertEqual({k: v[0] for k, v in RELEASED_2_1.items()}, hub)

    def test_vit_base_matches_the_hub_model_tensor_for_tensor(self):
        """Meta's loader does `load_state_dict(strict=True)` into this hub
        model, so equal keys and shapes here mean the released file fits."""
        hub_encoder, _ = backbones.vjepa2_1_vit_base_384(pretrained=False)
        ours = build_encoder(released_2_1_pretrain_config("vit_base"), 256, 16)
        self.assertEqual(_shapes(ours), _shapes(hub_encoder))
        self.assertEqual(len(_shapes(ours)), 158)

    def test_the_warm_start_arms_build_that_same_encoder(self):
        """The w-arms train with the proxy recipe's `model:` section; the file
        must fit THAT encoder, not only the probe's."""
        hub_encoder, _ = backbones.vjepa2_1_vit_base_384(pretrained=False)
        for name in ("w0-warm-control", "w2-warm-theta-temporal", "s1-sigreg"):
            cfg = yaml.safe_load((LADDER / f"{name}.yaml").read_text())
            with self.subTest(arm=name):
                self.assertEqual(_shapes(build_encoder(cfg, 256, 16)), _shapes(hub_encoder))

    def test_a_different_ffn_or_norm_does_not_fit(self):
        """The negative: an arm that changes the encoder's parameters cannot
        be warm-started, and the strict load must say so."""
        hub_encoder, _ = backbones.vjepa2_1_vit_base_384(pretrained=False)
        for name in ("f1-swiglu", "n1-dyt"):
            cfg = yaml.safe_load((LADDER / f"{name}.yaml").read_text())
            with self.subTest(arm=name):
                self.assertNotEqual(_shapes(build_encoder(cfg, 256, 16)), _shapes(hub_encoder))

    def test_unknown_architecture_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no released V-JEPA 2.1 model"):
            released_2_1_pretrain_config("vit_tiny")


class TestLoading(unittest.TestCase):
    def setUp(self):
        self.cfg = released_2_1_pretrain_config("vit_base")
        self.cfg["model"]["model_name"] = "vit_tiny"  # same code path, small

    def _file(self, tmp, key="ema_encoder", prefix="module.backbone."):
        torch.manual_seed(1)
        donor = build_encoder(self.cfg, 64, 4)
        path = Path(tmp) / "released.pt"
        torch.save({key: {prefix + k: v for k, v in donor.state_dict().items()}}, path)
        return path, donor

    def test_prefixes_are_stripped(self):
        cleaned = clean_encoder_state_dict(
            {"module.backbone.a.weight": 1, "backbone.b": 2, "c": 3, "module.d.backbone.e": 4}
        )
        self.assertEqual(set(cleaned), {"a.weight", "b", "c", "d.backbone.e"})

    def test_round_trip_into_a_bare_and_a_wrapped_encoder(self):
        from app.vjepa_3.wrappers import MultiSeqWrapper

        with tempfile.TemporaryDirectory() as tmp:
            path, donor = self._file(tmp)
            torch.manual_seed(2)
            bare = build_encoder(self.cfg, 64, 4)
            wrapped = MultiSeqWrapper(build_encoder(self.cfg, 64, 4))
            self.assertFalse(torch.equal(bare.patch_embed.proj.weight, donor.patch_embed.proj.weight))
            load_pretrained_encoder(bare, path)
            load_pretrained_encoder(wrapped, path)
            for loaded in (bare, wrapped.backbone):
                for k, v in donor.state_dict().items():
                    self.assertTrue(torch.equal(loaded.state_dict()[k], v), k)

    def test_missing_key_and_missing_tensor_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._file(tmp, key="target_encoder")
            with self.assertRaisesRegex(KeyError, "has no 'ema_encoder' entry"):
                read_encoder_state_dict(path)
            state = read_encoder_state_dict(path, "target_encoder")
            state.pop(next(iter(state)))
            torch.save({"ema_encoder": state}, path)
            with self.assertRaises(RuntimeError):
                load_pretrained_encoder(build_encoder(self.cfg, 64, 4), path)

    def test_a_file_that_is_not_plain_tensors_is_refused(self):
        """weights_only=True: a downloaded checkpoint must not run code."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evil.pt"
            torch.save({"ema_encoder": {"w": torch.zeros(1)}, "extra": Path("/tmp")}, path)
            with self.assertRaises(Exception) as caught:
                read_encoder_state_dict(path)
            self.assertIn("weights_only", str(caught.exception).lower().replace(" ", "_"))


if __name__ == "__main__":
    unittest.main()
