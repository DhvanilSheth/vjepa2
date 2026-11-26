# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""scripts/compare_tracking.py and scripts/preflight_arm.py."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = _load("compare_tracking")
preflight = _load("preflight_arm")


def _scores(shift=0.0, noise=0.0, seed=0, videos=30, per_video=2, video_shift=None):
    """Synthetic per-object J&F: a fixed base per object, plus a shift."""
    base_rng = np.random.default_rng(123)
    rng = np.random.default_rng(seed)
    scores = {}
    for v in range(videos):
        for obj in range(1, per_video + 1):
            value = base_rng.uniform(10, 80) + shift + rng.normal(0, noise)
            if video_shift and v in video_shift:
                value += video_shift[v]
            scores[(f"video{v:02d}", obj)] = value
    return scores


def _write(root, arm, scores, mode="frame"):
    path = Path(compare.result_path(str(root), arm, mode))
    path.parent.mkdir(parents=True)
    rows = [
        {"sequence": seq, "object": obj, "J": value / 100.0, "F": value / 100.0}
        for (seq, obj), value in scores.items()
    ]
    path.write_text(json.dumps({"per_object": rows}))


class TestCompareTracking(unittest.TestCase):
    def test_reads_the_probe_file_as_jf_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "a", {("x", 1): 40.0, ("x", 2): 60.0})
            scores = compare.load_scores(compare.result_path(tmp, "a", "frame"))
            self.assertEqual(scores, {("x", 1): 40.0, ("x", 2): 60.0})

    def test_identical_runs_have_no_difference(self):
        stats = compare.paired_bootstrap(_scores(), _scores())
        self.assertEqual((stats["diff"], stats["low"], stats["high"]), (0.0, 0.0, 0.0))
        self.assertEqual((stats["better"], stats["worse"], stats["videos"]), (0, 0, 30))

    def test_a_uniform_gain_is_found_and_a_noisy_one_is_not_overclaimed(self):
        control = _scores()
        clean = compare.paired_bootstrap(_scores(shift=2.0), control)
        self.assertAlmostEqual(clean["diff"], 2.0, places=6)
        self.assertGreater(clean["low"], 0)
        noisy = compare.paired_bootstrap(_scores(shift=0.5, noise=6.0, seed=4), control)
        self.assertTrue(noisy["low"] < 0 < noisy["high"], noisy)

    def test_resampling_is_by_video_not_by_object(self):
        """One video with eight objects carries the whole gain.  Resampling
        objects would treat those eight as independent evidence; resampling
        videos must leave zero inside the range."""
        control = _scores(videos=30, per_video=1)
        arm = dict(control)
        for obj in range(2, 10):
            control[("video00", obj)] = 30.0
            arm[("video00", obj)] = 60.0
        stats = compare.paired_bootstrap(arm, control)
        self.assertGreater(stats["diff"], 5.0)
        self.assertEqual(stats["low"], 0.0)

    def test_verdict_needs_both_kinds_of_noise_cleared(self):
        stats = {"diff": 1.4, "low": 0.1, "high": 2.9}
        self.assertEqual(compare.verdict({"diff": 0.3, "low": -0.6, "high": 1.2}, 1.0), "not separable")
        self.assertIn("training noise unknown", compare.verdict(stats, None))
        self.assertEqual(compare.verdict(stats, 2.0), "within training noise")
        self.assertEqual(compare.verdict(stats, 0.5), "clear: better")
        self.assertEqual(compare.verdict({"diff": -3, "low": -4, "high": -2}, 0.5), "clear: worse")

    def test_seed_gap_is_the_largest_pairwise_gap(self):
        a, b, c = _scores(), _scores(shift=0.8), _scores(shift=-0.4)
        self.assertIsNone(compare.seed_gap([a]))
        self.assertAlmostEqual(compare.seed_gap([a, b]), 0.8, places=6)
        self.assertAlmostEqual(compare.seed_gap([a, b, c]), 1.2, places=6)

    def test_different_object_sets_are_refused(self):
        other = _scores()
        other.pop(next(iter(other)))
        with self.assertRaisesRegex(ValueError, "different objects"):
            compare.paired_bootstrap(other, _scores())

    def test_command_line_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write(tmp, "a0-control", _scores())
            _write(tmp, "a0-control-s2", _scores(shift=0.6))
            _write(tmp, "good", _scores(shift=3.0))
            _write(tmp, "meh", _scores(shift=0.3, noise=6.0, seed=2))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = compare.main(["--runs", tmp, "--control", "a0-control", "--control", "a0-control-s2"])
            text = out.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("largest gap between controls = 0.6", text)
            lines = {line.split()[0]: line for line in text.splitlines() if line[:4] in ("good", "meh ")}
            self.assertIn("clear: better", lines["good"])
            self.assertNotIn("clear", lines["meh"])
            self.assertNotIn("\na0-control ", text.split("verdict")[1])


class TestPreflight(unittest.TestCase):
    """At a tiny model size, so this is a test of the script, not of the arms;
    the same contrast holds at ViT-B (V3_STRING/PROGRESS.md)."""

    @classmethod
    def setUpClass(cls):
        ladder = REPO_ROOT / "configs" / "train_3" / "ladder"
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.control = preflight.one_step(ladder / "a0-control.yaml", tmp / "c", tiny=True)
        cls.arms = {
            name: preflight.one_step(ladder / f"{name}.yaml", tmp / name, tiny=True)
            for name in ("n1-dyt", "p2-theta-temporal", "s1-sigreg")
        }

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_it_flags_the_arm_that_collapsed(self):
        rows, problems = preflight.check(self.arms["n1-dyt"], self.control)
        self.assertTrue(any("encoder gradient is" in p for p in problems), problems)
        ratio = {row[0]: row[2] for row in rows}
        self.assertLess(ratio["encoder"], 1 / preflight.RATIO_LIMIT)

    def test_it_passes_a_healthy_arm(self):
        for name in ("p2-theta-temporal", "s1-sigreg"):
            with self.subTest(arm=name):
                _, problems = preflight.check(self.arms[name], self.control)
                self.assertEqual(problems, [])

    def test_new_share_isolates_the_regulariser(self):
        """Same start, same data: the gradient difference IS the regulariser's
        gradient -- present on the encoder, exactly absent on the predictor."""
        rows = {row[0]: row for row in preflight.check(self.arms["s1-sigreg"], self.control)[0]}
        self.assertGreater(rows["encoder"][5], 0.05)
        self.assertEqual(rows["predictor"][5], 0.0)
        # an arm with different parameters has no comparable start
        dyt = {row[0]: row for row in preflight.check(self.arms["n1-dyt"], self.control)[0]}
        self.assertIsNone(dyt["encoder"][5])

    def test_the_control_against_itself_is_a_clean_pass(self):
        rows, problems = preflight.check(self.control, self.control)
        self.assertEqual(problems, [])
        for row in rows:
            self.assertEqual(row[2], 1.0)
            self.assertEqual(row[5], 0.0)

    def test_exit_status_and_skip_of_arms_without_their_file(self):
        ladder = REPO_ROOT / "configs" / "train_3" / "ladder"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            bad = preflight.main(["--tiny", str(ladder / "n1-dyt.yaml")])
            good = preflight.main(["--tiny", str(ladder / "s2-visreg.yaml"), str(ladder / "w0-warm-control.yaml")])
        self.assertEqual((bad, good), (1, 0))
        self.assertIn("w0-warm-control: SKIP", out.getvalue())


if __name__ == "__main__":
    unittest.main()
