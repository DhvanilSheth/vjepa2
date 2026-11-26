# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The real `app/vjepa_3/train.py:main()`, end to end, on CPU.

Before this file no test ran the training loop at all (ladder audit, C-22 and
L-26).  Each switch added to the loop is tested here for three things:

  * OFF  -- the default run is still V-JEPA 2.1's loop, bit for bit;
  * ON   -- the switch changes what it says it changes;
  * ONLY -- it changes nothing else (where that can be stated exactly).

See tests/train/harness.py for the two things that are substituted.
"""

import csv
import os
import tempfile
import unittest
from pathlib import Path

import torch

from app.vjepa_3.pretrained import clean_encoder_state_dict as bare
from tests.train.harness import max_abs_diff, run_train, states_equal, tiny_config

MODELS = ("encoder", "predictor", "target_encoder")


def _delta_norm(after, before):
    keys = [k for k in after if after[k].is_floating_point()]
    return sum(float((after[k] - before[k]).double().square().sum()) for k in keys) ** 0.5


class TrainLoopCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def cfg(self, name, **overrides):
        return tiny_config(self.root / name, **overrides)


class TestDefaultsAreV21(TrainLoopCase):
    def test_default_run_equals_the_2_1_loop_bit_for_bit(self):
        """Every v3 key at its default: the two training loops must produce the
        same checkpoint.  ViT-L, because V-JEPA 2.1's loop cannot train a
        smaller model (it only knows the widths of L, g and G).
        """
        common = dict(model_name="vit_large", epochs=1, ipe=2, optimization={"lambda_warmup_frac": None})
        v3 = run_train(self.cfg("v3", **common))
        v21 = run_train(self.cfg("v21", **common), app="vjepa_2_1")
        for model in MODELS:
            with self.subTest(model=model):
                self.assertTrue(states_equal(v3["latest"][model], v21["latest"][model]))
        self.assertEqual(v3["latest"]["loss"], v21["latest"]["loss"])

    def test_two_runs_of_one_config_are_identical(self):
        """The property every "this switch changed X" test below leans on."""
        a = run_train(self.cfg("a"))
        b = run_train(self.cfg("b"))
        for model in MODELS:
            self.assertTrue(states_equal(a["latest"][model], b["latest"][model]))

    def test_default_run_writes_no_regulariser_log(self):
        run_train(self.cfg("a"))
        self.assertEqual(
            sorted(os.listdir(self.root / "a")),
            ["e0.pth.tar", "e1.pth.tar", "latest.pth.tar", "log_r0.csv"],
        )


class TestRegulariser(TrainLoopCase):
    def _reg(self, name, reg_type, weight=0.5, **kw):
        loss = {"reg_type": reg_type, "reg_weight": weight, "reg_num_slices": 64, "reg_max_tokens": 256}
        return self.cfg(name, loss=loss, **kw)

    def test_each_regulariser_trains_and_logs_both_parts(self):
        control = run_train(self.cfg("control"))
        for reg_type in ("sigreg", "visreg"):
            with self.subTest(reg=reg_type):
                out = run_train(self._reg(reg_type, reg_type))
                self.assertFalse(states_equal(out["latest"]["encoder"], control["latest"]["encoder"]))
                for model in MODELS:
                    for value in out["latest"][model].values():
                        self.assertTrue(torch.isfinite(value).all())
                with open(self.root / reg_type / "reg_r0.csv") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), out["steps"])
                self.assertTrue(all(float(r["loss_reg"]) > 0 for r in rows))
                # iteration 0 sees the same weights and data as the control, so
                # the logged main loss must be the control's first loss
                with open(self.root / "control" / "log_r0.csv") as f:
                    first = next(csv.DictReader(f))
                self.assertAlmostEqual(float(rows[0]["loss_main"]), float(first["loss"]), places=4)

    def test_it_acts_on_the_encoder_only(self):
        """One step: the regulariser reads encoder tokens, so the predictor's
        gradient -- and with it the predictor's first update -- is untouched."""
        one_step = dict(epochs=1, ipe=1)
        control = run_train(self.cfg("control", **one_step))
        out = run_train(self._reg("reg", "visreg", **one_step))
        self.assertTrue(states_equal(out["latest"]["predictor"], control["latest"]["predictor"]))
        self.assertFalse(states_equal(out["latest"]["encoder"], control["latest"]["encoder"]))

    def test_a_larger_weight_lowers_the_regulariser_loss(self):
        """The term must actually be optimised, not just added and logged."""

        def final_reg(name, weight):
            run_train(self._reg(name, "visreg", weight=weight, epochs=1, ipe=12, optimization={"lr": 0.001, "start_lr": 0.001, "final_lr": 0.001}))
            with open(self.root / name / "reg_r0.csv") as f:
                return float(list(csv.DictReader(f))[-1]["loss_reg"])

        self.assertLess(final_reg("strong", 5.0), final_reg("weak", 1e-6))

    def test_half_configured_regulariser_is_refused(self):
        for loss in ({"reg_type": "sigreg", "reg_weight": 0.0}, {"reg_type": None, "reg_weight": 0.5}):
            with self.subTest(loss=loss), self.assertRaises(ValueError):
                run_train(self.cfg(f"bad{loss['reg_weight']}", loss=loss))


class TestWarmStart(TrainLoopCase):
    def _released_file(self, key="ema_encoder"):
        """A file laid out like Meta's: weights different from any seeded init."""
        donor = run_train(self.cfg("donor", meta={"seed": 77}, epochs=1, ipe=2))
        weights = {k: v.clone() for k, v in donor["latest"]["target_encoder"].items()}
        path = self.root / "released.pt"
        torch.save({key: weights, "predictor": {"unrelated": torch.zeros(1)}}, path)
        # `bare` strips the wrapper prefixes: the file and a checkpoint carry
        # `module.backbone.`, the pre-step snapshot only `backbone.`
        return path, bare(weights)

    def test_encoder_and_target_start_from_the_file_predictor_does_not(self):
        path, weights = self._released_file()
        scratch = run_train(self.cfg("scratch", epochs=1, ipe=1))
        warm = run_train(self.cfg("warm", epochs=1, ipe=1, meta={"pretrained_encoder": str(path)}))
        self.assertTrue(states_equal(bare(warm["init_encoder"]), weights))
        self.assertFalse(states_equal(bare(scratch["init_encoder"]), weights))
        self.assertTrue(states_equal(warm["init_predictor"], scratch["init_predictor"]))
        # the EMA target after one step is still within one EMA step of the file
        self.assertLess(max_abs_diff(bare(warm["latest"]["target_encoder"]), weights), 1e-5)
        self.assertGreater(max_abs_diff(bare(scratch["latest"]["target_encoder"]), weights), 1e-3)

    def test_wrong_key_or_wrong_architecture_is_refused(self):
        path, _ = self._released_file()
        with self.assertRaisesRegex(KeyError, "no 'target_encoder' entry"):
            run_train(self.cfg("k", meta={"pretrained_encoder": str(path), "pretrained_encoder_key": "target_encoder"}))
        with self.assertRaises(RuntimeError):  # strict load: vit_small != vit_tiny
            run_train(self.cfg("a", model_name="vit_small", meta={"pretrained_encoder": str(path)}))

    def test_frozen_encoder_does_not_move_then_does(self):
        path, weights = self._released_file()
        warm = {"pretrained_encoder": str(path)}
        frozen = run_train(self.cfg("frozen", epochs=1, ipe=3, meta=warm, optimization={"freeze_encoder_iters": 3}))
        for model in ("encoder", "target_encoder"):
            self.assertTrue(states_equal(bare(frozen["latest"][model]), weights), model)
        self.assertFalse(
            states_equal(bare(frozen["latest"]["predictor"]), bare(frozen["init_predictor"]))
        )

        thawed = run_train(self.cfg("thawed", epochs=1, ipe=3, meta=warm, optimization={"freeze_encoder_iters": 2}))
        self.assertFalse(states_equal(bare(thawed["latest"]["encoder"]), weights))
        # one step of movement instead of three
        free = run_train(self.cfg("free", epochs=1, ipe=3, meta=warm))
        self.assertGreater(
            _delta_norm(bare(free["latest"]["encoder"]), weights),
            _delta_norm(bare(thawed["latest"]["encoder"]), weights),
        )

    def test_encoder_lr_scale_scales_the_encoder_only(self):
        """Adam's first update has size ~lr per element, so a tenth of the
        learning rate must give a tenth of the movement; the predictor's first
        step must not change at all."""
        base = run_train(self.cfg("base", epochs=1, ipe=1))
        slow = run_train(self.cfg("slow", epochs=1, ipe=1, optimization={"encoder_lr_scale": 0.1}))
        self.assertTrue(states_equal(slow["latest"]["predictor"], base["latest"]["predictor"]))
        ratio = _delta_norm(bare(slow["latest"]["encoder"]), bare(slow["init_encoder"])) / _delta_norm(
            bare(base["latest"]["encoder"]), bare(base["init_encoder"])
        )
        self.assertAlmostEqual(ratio, 0.1, delta=0.005)

    def test_resume_overrides_the_pretrained_file(self):
        path, weights = self._released_file()
        cfg = self.cfg("run", epochs=2, ipe=2, meta={"pretrained_encoder": str(path)})

        def crash(step):
            if step == 2:
                raise KeyboardInterrupt("simulated crash at the start of epoch 2")

        with self.assertRaises(KeyboardInterrupt):
            run_train(cfg, before_step=crash)
        saved = torch.load(self.root / "run" / "latest.pth.tar", weights_only=False)
        self.assertEqual(saved["epoch"], 1)
        resumed = run_train(cfg)
        self.assertEqual(resumed["latest"]["epoch"], 2)
        self.assertEqual(resumed["steps"], 2, "the resumed run must do only the second epoch")
        # it continued from the epoch-1 weights, not from the file again
        self.assertTrue(states_equal(bare(resumed["init_encoder"]), bare(saved["encoder"])))
        self.assertFalse(states_equal(bare(resumed["init_encoder"]), weights))


class TestTimeFocusedArms(TrainLoopCase):
    def test_future_prediction_mask_trains_through_the_real_loop(self):
        """mask.0.max_temporal_keep < 1 changes what the encoder sees, so the
        run must differ from the control and stay finite."""
        cfg = self.cfg("future")
        cfg["mask"][0]["max_temporal_keep"] = 0.5
        out = run_train(cfg)
        control = run_train(self.cfg("control"))
        self.assertFalse(states_equal(out["latest"]["encoder"], control["latest"]["encoder"]))
        for model in MODELS:
            for value in out["latest"][model].values():
                self.assertTrue(torch.isfinite(value).all())

    def test_whole_frame_and_whole_clip_masks_train_through_the_real_loop(self):
        """The tiny clip has 4 time-steps, so 2 are hidden here (the arm
        hides 6 of 8).  Each option must change the run and keep it finite."""
        control = run_train(self.cfg("control"))
        for name, edit in (("frames", {"hide_time_steps": 2}), ("whole", {"truncation": "random"})):
            cfg = self.cfg(name)
            cfg["mask"][0].update(edit)
            out = run_train(cfg)
            with self.subTest(option=name):
                self.assertFalse(states_equal(out["latest"]["encoder"], control["latest"]["encoder"]))
                for model in MODELS:
                    for value in out["latest"][model].values():
                        self.assertTrue(torch.isfinite(value).all())

    def test_fps_reaches_the_loader(self):
        out = run_train(self.cfg("fps", epochs=1, ipe=1, data={"fps": 12}))
        self.assertEqual(out["init_data_kwargs"]["fps"], 12)


class TestClipGrad(TrainLoopCase):
    def _norms(self, name, **optimization):
        norms = []

        def record(encoder, predictor):
            grads = [p.grad for m in (encoder, predictor) for p in m.parameters() if p.grad is not None]
            norms.append(float(torch.cat([g.flatten() for g in grads]).norm()))

        run_train(self.cfg(name, epochs=1, ipe=2, optimization=optimization), on_optimizer_step=record)
        return norms

    def test_gradient_norm_is_capped_only_when_asked(self):
        free = self._norms("free")
        cap = min(free) / 10
        clipped = self._norms("clipped", clip_grad=cap)
        self.assertTrue(all(n > cap * 5 for n in free))
        for n in clipped:
            self.assertLessEqual(n, cap * 1.001)

    def test_bad_value_is_refused(self):
        with self.assertRaises(ValueError):
            run_train(self.cfg("bad", optimization={"clip_grad": 0}))


class TestCheckpointing(TrainLoopCase):
    def test_no_temporary_file_is_left_behind(self):
        run_train(self.cfg("a"))
        self.assertEqual([f for f in os.listdir(self.root / "a") if f.endswith(".tmp")], [])

    def test_failed_write_keeps_the_previous_checkpoint_and_fails_the_run(self):
        """V-JEPA 2.1 wrote latest.pth.tar in place and only logged a failed
        save, so a full disk could end a run with a truncated file and exit
        status 0 (ladder audit L-10)."""
        real_save, calls = torch.save, {"n": 0}

        def flaky_save(obj, path, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 2:  # epoch 1 writes latest + e0; fail everything after
                with open(path, "wb") as f:
                    f.write(b"truncated")
                raise OSError("No space left on device")
            return real_save(obj, path, *args, **kwargs)

        torch.save = flaky_save
        try:
            with self.assertRaisesRegex(RuntimeError, "final checkpoint could not be written"):
                run_train(self.cfg("a"))
        finally:
            torch.save = real_save
        folder = self.root / "a"
        latest = torch.load(folder / "latest.pth.tar", weights_only=False)
        self.assertEqual(latest["epoch"], 1, "the epoch-1 checkpoint must survive intact")
        self.assertEqual([f for f in os.listdir(folder) if f.endswith(".tmp")], [])
        self.assertFalse((folder / "e1.pth.tar").exists())


class TestDataLoaderOptions(TrainLoopCase):
    def test_persistent_workers_reaches_the_loader(self):
        for value in (True, False):
            with self.subTest(value=value):
                out = run_train(self.cfg(f"p{value}", epochs=1, ipe=1, data={"persistent_workers": value}))
                self.assertIs(out["init_data_kwargs"]["persistent_workers"], value)

    def test_absent_key_keeps_the_2_1_behaviour(self):
        cfg = self.cfg("absent", epochs=1, ipe=1)
        del cfg["data"]["persistent_workers"]
        self.assertIs(run_train(cfg)["init_data_kwargs"]["persistent_workers"], False)


if __name__ == "__main__":
    unittest.main()
