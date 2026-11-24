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

import importlib.util
import re
import unittest
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "train_3"
TRAIN_PY = REPO_ROOT / "app" / "vjepa_3" / "train.py"
LADDER_DIR = CONFIG_DIR / "ladder"
PROXY_BASE = CONFIG_DIR / "proxy" / "base-1gpu.yaml"
LADDER_SCRIPT = REPO_ROOT / "scripts" / "make_ladder_configs.py"

# Every V-JEPA 3 `model:` knob and the value that reproduces V-JEPA 2.1.
V3_MODEL_DEFAULTS = {
    "rope_theta": 10000.0,
    "rope_temporal_scale": 1.0,
    "use_string": False,
    "use_string_predictor": False,
    "string_share_across_layers": False,
    "string_init_std": 0.0,
    "ffn_type": None,
    "pred_ffn_type": None,
    "norm_type": None,
    "norm_out_type": None,
}

# `meta:` keys present in the configs that nothing reads.  Inherited from
# V-JEPA 2.1 and listed so the meta-key test can still catch a NEW orphan.
KNOWN_DEAD_META_KEYS = {
    # configs/train_2_1/**.yaml ship `eval_freq`; no reader in app/ or src/.
    "eval_freq",
}


def _baseline_configs():
    """Configs that must sit at V-JEPA 2.1 behaviour: everything except the
    ladder arms, whose whole purpose is to move one knob off baseline."""
    return [p for p in sorted(CONFIG_DIR.rglob("*.yaml")) if LADDER_DIR not in p.parents]


def _load_ladder_script():
    spec = importlib.util.spec_from_file_location("make_ladder_configs", LADDER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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
        expected = V3_MODEL_DEFAULTS
        for path in _baseline_configs():
            cfg = yaml.safe_load(path.read_text())
            with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                self.assertEqual(cfg["app"], "vjepa_3")
                for key, default in expected.items():
                    self.assertIn(key, cfg["model"], f"{key} missing")
                    self.assertEqual(cfg["model"][key], default, f"{key} not at baseline")

    def test_every_v3_key_is_forwarded_to_init_video_model(self):
        """Reading a key is hop 1; it must also be PASSED ON (hop 2).

        `x = cfgs_model.get("x")` followed by forgetting `x=x` in the
        init_video_model(...) call is the same silent no-op one hop later.
        """
        source = TRAIN_PY.read_text()
        call = source[source.index("init_video_model(\n") :]
        call = call[: call.index("target_encoder = copy.deepcopy(encoder)")]
        for key in V3_MODEL_DEFAULTS:
            with self.subTest(key=key):
                self.assertIn(f"{key}={key},", call)

    def test_every_meta_key_is_read(self):
        source = TRAIN_PY.read_text()
        read_keys = set(re.findall(r'cfgs_meta\.get\(\s*"([^"]+)"', source))
        orphans = {}
        for path in sorted(CONFIG_DIR.rglob("*.yaml")):
            cfg = yaml.safe_load(path.read_text())
            for key in cfg.get("meta") or {}:
                if key not in read_keys | KNOWN_DEAD_META_KEYS:
                    orphans.setdefault(key, []).append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(orphans, {}, "meta keys with no reader in train.py")
        self.assertIn("sigreg_diag_freq", read_keys)

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


class TestSingleGpuProxyBase(unittest.TestCase):
    """configs/train_3/proxy/base-1gpu.yaml must actually be a 1-GPU config."""

    def setUp(self):
        self.cfg = yaml.safe_load(PROXY_BASE.read_text())

    def test_single_rank_and_video_only(self):
        self.assertEqual(self.cfg["nodes"] * self.cfg["tasks_per_node"], 1)
        # train.py raises on img_data with a single rank (images and video
        # train on disjoint ranks), so the base must not declare it.
        self.assertNotIn("img_data", self.cfg)
        self.assertNotIn("img_mask", self.cfg)

    def test_train_py_guards_the_single_rank_image_split(self):
        """The guard the comment above relies on must exist, BEFORE the modulo
        that used to divide by zero."""
        source = TRAIN_PY.read_text()
        guard = source.index("if img_world_size == 0 or num_video_ranks == 0:")
        modulo = source.index("img_total_batch_size % img_world_size")
        self.assertLess(guard, modulo)

    def test_differs_from_the_paper_recipe_only_where_documented(self):
        paper = yaml.safe_load(
            (CONFIG_DIR / "vitb16" / "pretrain-256px-16f.yaml").read_text()
        )
        # The one documented `model:` difference: activation checkpointing
        # is off (it trades compute for memory the proxy does not need).  It
        # changes no parameter and no gradient.
        model, paper_model = dict(self.cfg["model"]), dict(paper["model"])
        self.assertFalse(model.pop("use_activation_checkpointing"))
        self.assertTrue(paper_model.pop("use_activation_checkpointing"))
        self.assertEqual(model, paper_model)
        self.assertEqual(self.cfg["mask"], paper["mask"])
        self.assertEqual(self.cfg["loss"], paper["loss"])
        self.assertEqual(self.cfg["data_aug"], paper["data_aug"])


class TestContextLossWarmup(unittest.TestCase):
    """The context ("dense") loss must actually switch on inside a run.

    V-JEPA 2.1 hard-codes its warm-up at iterations 15,000 -> 30,000, 5% ->
    10% of a 300,000-iteration recipe.  A proxy run is far shorter, so the
    absolute numbers would leave the loss off for most or all of it.
    """

    def test_default_is_the_2_1_schedule(self):
        from app.vjepa_3.models.utils.modules import Lambda_LinearWarmupHold

        sched = Lambda_LinearWarmupHold(lambda_value=0.5)
        self.assertEqual((sched.start, sched.end), (15_000, 30_000))
        source = TRAIN_PY.read_text()
        self.assertIn('cfgs_opt.get("lambda_warmup_frac", None)', source)
        for path in _baseline_configs():
            if path == PROXY_BASE:
                continue
            cfg = yaml.safe_load(path.read_text())
            with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                self.assertNotIn("lambda_warmup_frac", cfg["optimization"])

    def test_paper_fractions_reproduce_the_paper_iterations(self):
        from app.vjepa_3.utils import lambda_warmup_iters

        self.assertEqual(lambda_warmup_iters([0.05, 0.1], 300_000), (15_000, 30_000))
        self.assertEqual(lambda_warmup_iters([0.05, 0.1], 9_000), (450, 900))
        for bad in ([0.1], [0.2, 0.1], [0.1, 0.1], [-0.1, 0.5], [0.5, 1.5]):
            with self.subTest(frac=bad), self.assertRaises(ValueError):
                lambda_warmup_iters(bad, 9_000)

    def test_proxy_run_spends_most_of_its_length_at_full_weight(self):
        from app.vjepa_3.utils import lambda_warmup_iters

        opt = yaml.safe_load(PROXY_BASE.read_text())["optimization"]
        total = opt["epochs"] * opt["ipe"]
        self.assertIn("lambda_warmup_frac", opt, "proxy would use 15,000 -> 30,000")
        _, end = lambda_warmup_iters(opt["lambda_warmup_frac"], total)
        self.assertLessEqual(end, 0.25 * total)
        # ...and without the key this run would never reach full weight.
        self.assertLess(total, 30_000)


class TestAblationLadder(unittest.TestCase):
    """configs/train_3/ladder/ is generated; these keep it honest."""

    @classmethod
    def setUpClass(cls):
        cls.script = _load_ladder_script()
        cls.base = yaml.safe_load(PROXY_BASE.read_text())

    def test_files_on_disk_match_the_generator(self):
        files = self.script.build_all()
        self.assertGreater(len(files), 1)
        for path, text in files.items():
            with self.subTest(arm=path.name):
                self.assertTrue(path.exists(), "run scripts/make_ladder_configs.py")
                self.assertEqual(path.read_text(), text, "stale: re-run the generator")
        on_disk = {p.name for p in LADDER_DIR.glob("*.yaml")}
        self.assertEqual(on_disk, {p.name for p in files}, "orphan ladder file")
        self.assertEqual(self.script.main(["--check"]), 0)

    def test_each_arm_changes_only_what_it_declares(self):
        for name, (_, _, overrides) in self.script.ARMS.items():
            with self.subTest(arm=name):
                cfg = yaml.safe_load((LADDER_DIR / f"{name}.yaml").read_text())
                self.assertTrue(cfg.pop("folder").endswith(f"/{name}"))
                base = {k: v for k, v in self.base.items() if k != "folder"}
                model, base_model = cfg.pop("model"), base.pop("model")
                self.assertEqual(cfg, base, "an arm changed something outside model:")
                diff = {k: model[k] for k in model if model[k] != base_model.get(k)}
                self.assertEqual(diff, overrides)
                self.assertEqual(set(model), set(base_model))

    def test_control_is_the_baseline_and_arms_are_distinct(self):
        arms = self.script.ARMS
        self.assertEqual(arms["a0-control"][2], {})
        for key, default in V3_MODEL_DEFAULTS.items():
            self.assertEqual(self.base["model"][key], default)
        overrides = [tuple(sorted(map(str, spec[2].items()))) for spec in arms.values()]
        self.assertEqual(len(set(overrides)), len(overrides), "two arms are identical")
        for name, (_, _, changed) in arms.items():
            self.assertLessEqual(set(changed), set(V3_MODEL_DEFAULTS), name)

    def test_warm_start_flag_matches_what_the_arm_changes(self):
        """FFN / norm arms change the function the weights compute, so they
        cannot be initialised from a V-JEPA 2.1 checkpoint."""
        from_scratch_keys = {"ffn_type", "pred_ffn_type", "norm_type", "norm_out_type"}
        for name, (_, warm, changed) in self.script.ARMS.items():
            with self.subTest(arm=name):
                self.assertEqual(warm, not (set(changed) & from_scratch_keys))

    def test_every_arm_builds_and_runs(self):
        """Build each arm's encoder (at vit_tiny width, CPU) and run it."""
        from evals.feature_clustering_frozen.eval import build_encoder

        torch.manual_seed(0)
        x = torch.randn(1, 3, 8, 64, 64)
        outputs = {}
        for name in self.script.ARMS:
            cfg = yaml.safe_load((LADDER_DIR / f"{name}.yaml").read_text())
            cfg["model"]["model_name"] = "vit_tiny"
            with self.subTest(arm=name):
                torch.manual_seed(0)
                enc = build_encoder(cfg, resolution=64, frames_per_clip=8).eval()
                with torch.no_grad():
                    outputs[name] = enc(x)
                self.assertTrue(torch.isfinite(outputs[name]).all())
        # p3-string starts at P = I, so it is SUPPOSED to equal the control.
        torch.testing.assert_close(outputs["p3-string"], outputs["a0-control"], rtol=0, atol=0)
        for name, out in outputs.items():
            if name not in ("a0-control", "p3-string"):
                with self.subTest(arm=name):
                    self.assertFalse(
                        torch.allclose(out, outputs["a0-control"]),
                        f"{name} is indistinguishable from the control",
                    )

    def test_preset_merge_rejects_conflicts(self):
        script = _load_ladder_script()  # private copy: we mutate PRESET
        script.PRESET = ["n1-dyt", "n2-derf"]
        with self.assertRaisesRegex(ValueError, "disagree on `norm_type`"):
            script.build_all()
        script.PRESET = ["p2-theta-temporal", "n2-derf"]
        files = script.build_all()
        merged = yaml.safe_load(files[LADDER_DIR / "modded.yaml"])["model"]
        self.assertEqual(merged["norm_type"], "derf")
        self.assertEqual(merged["rope_theta"], [100.0, 10000.0, 10000.0])


class TestRegistersAreNotImplemented(unittest.TestCase):
    """`n_registers` looks like a feature switch.  It is not one.

    Neither V-JEPA 2.1 nor this fork ever CREATES register tokens: there is
    no parameter and no concatenation anywhere.  The key only tells RoPE to
    treat the last `n` tokens of the sequence as position-free -- and those
    are ordinary patch tokens.  On the encoder it does not even get that far:
    the rotary table is built for the full T*H*W grid while the tokens it is
    applied to have been cut to N - n, so the forward pass raises.

    This pins both facts so the key is not used as an ablation arm by
    mistake.  If real register tokens are ever implemented, these tests
    should fail and be rewritten.
    """

    KWARGS = dict(img_size=(64, 64), num_frames=8, use_rope=True, img_temporal_dim_size=1)

    def test_n_registers_adds_no_parameters(self):
        import app.vjepa_3.models.vision_transformer as vit_3

        plain = vit_3.vit_tiny(**self.KWARGS)
        with_reg = vit_3.vit_tiny(**self.KWARGS, n_registers=4)
        self.assertEqual(sorted(plain.state_dict()), sorted(with_reg.state_dict()))

    def test_n_registers_crashes_the_encoder_forward(self):
        import app.vjepa_2_1.models.vision_transformer as vit_2_1
        import app.vjepa_3.models.vision_transformer as vit_3

        x = torch.randn(1, 3, 8, 64, 64)
        for tree in (vit_2_1, vit_3):  # inherited from 2.1, not introduced here
            with self.subTest(tree=tree.__name__):
                model = tree.vit_tiny(**self.KWARGS, n_registers=4).eval()
                with self.assertRaisesRegex(RuntimeError, "must match the size"):
                    with torch.no_grad():
                        model(x)

    def test_no_shipped_config_enables_it(self):
        for path in sorted(CONFIG_DIR.rglob("*.yaml")):
            model = yaml.safe_load(path.read_text())["model"]
            with self.subTest(config=str(path.relative_to(REPO_ROOT))):
                self.assertEqual(model.get("n_registers", 0), 0)
                self.assertEqual(model.get("n_registers_predictor", 0), 0)


if __name__ == "__main__":
    unittest.main()
