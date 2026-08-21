# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Every `model:` key in a V-JEPA 3 config must have a reader in train.py.

WHY THIS EXISTS
---------------
`VisionTransformer.__init__`, `VisionTransformerPredictor.__init__` and
`Block.__init__` all end in `**kwargs`, which swallows unknown keyword
arguments with no error.  A yaml key that is never read by
`cfgs_model.get(...)` in train.py therefore does exactly nothing, silently,
and looks configured.

That is not hypothetical.  V-JEPA 2.1 ships three live instances:
  * `local_window`, `pred_local_window`, `vit_conv` in
    configs/train_2_1/**.yaml -- no reader anywhere in app/ or src/
  * `grid_depth` passed to Block() from vision_transformer.py:129 and
    predictor.py:161 -- Block has no such parameter
  * `use_SiLU` / `wide_SiLU` at src/hub/backbones.py:232 -- miscased against
    the real `use_silu` / `wide_silu`

This test fails the build rather than letting a fourth join them.
"""

import re
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "train_3"
TRAIN_PY = REPO_ROOT / "app" / "vjepa_3" / "train.py"

# Keys consumed somewhere other than a `cfgs_model.get(...)` in train.py.
# Keep this list SHORT and justified -- it is an escape hatch, not a dumping
# ground.  Each entry is a key that is read, just not through the usual hop.
KNOWN_NON_CFGS_MODEL_READERS = {
    # Read via kwargs.get() inside VisionTransformerPredictor.__init__
    # (predictor.py:90) rather than in train.py.
    "levels_predictor",
}


def _model_keys_in_configs():
    keys = {}
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        for key in (cfg.get("model") or {}):
            keys.setdefault(key, []).append(path.relative_to(REPO_ROOT))
    return keys


class TestVJepa3ConfigKeysHaveReaders(unittest.TestCase):
    def test_configs_exist(self):
        self.assertTrue(CONFIG_DIR.is_dir(), f"missing {CONFIG_DIR}")
        self.assertGreater(len(list(CONFIG_DIR.rglob("*.yaml"))), 0)

    def test_every_model_key_is_read(self):
        source = TRAIN_PY.read_text()
        read_keys = set(re.findall(r'cfgs_model\.get\(\s*"([^"]+)"', source))
        read_keys |= KNOWN_NON_CFGS_MODEL_READERS

        orphans = {
            key: paths
            for key, paths in _model_keys_in_configs().items()
            if key not in read_keys
        }
        self.assertEqual(
            orphans,
            {},
            "These configs/train_3 `model:` keys have no reader in "
            "app/vjepa_3/train.py, so they silently do nothing:\n"
            + "\n".join(f"  {k}: {[str(p) for p in v]}" for k, v in sorted(orphans.items())),
        )

    def test_vjepa3_keys_are_present_and_defaulted_to_baseline(self):
        """Every v3 config must ship the new knobs, set to V-JEPA 2.1 behaviour."""
        expected = {
            "rope_theta": 10000.0,
            "rope_temporal_scale": 1.0,
            "use_string": False,
            "use_string_predictor": False,
            "string_share_across_layers": False,
            "string_init_std": 0.0,
        }
        for path in sorted(CONFIG_DIR.rglob("*.yaml")):
            cfg = yaml.safe_load(path.read_text())
            with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                self.assertEqual(cfg["app"], "vjepa_3")
                for key, default in expected.items():
                    self.assertIn(key, cfg["model"], f"{key} missing")
                    self.assertEqual(cfg["model"][key], default, f"{key} not at baseline")

    def test_baseline_configs_declare_no_orphan_from_2_1(self):
        """The dead 2.1 keys must not be carried into v3 configs.

        `local_window`, `pred_local_window` and `vit_conv` appear in
        configs/train_2_1/**.yaml and are read by nothing.  If they are still
        present here, they were copied without being audited.
        """
        dead = {"local_window", "pred_local_window", "vit_conv"}
        for path in sorted(CONFIG_DIR.rglob("*.yaml")):
            cfg = yaml.safe_load(path.read_text())
            with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                self.assertEqual(dead & set(cfg["model"]), set())


if __name__ == "__main__":
    unittest.main()
