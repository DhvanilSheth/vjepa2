# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""scripts/measure_change.py: the change signal on scenes with a known answer,
and the whole script on a small fake DAVIS folder."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "measure_change", REPO_ROOT / "scripts" / "measure_change.py"
)
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)

R, P = mc.SIZE, mc.PATCH
G = R // P


def _blur(x, passes):
    for _ in range(passes):
        x = (x + np.roll(x, 1, 0) + np.roll(x, -1, 0) + np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 5
    return x


def _texture(seed):
    """A [R, R] image in [0, 1] with detail at every scale down to a few pixels."""
    rng = np.random.default_rng(seed)
    coarse = _blur(rng.standard_normal((R, R)), 60)
    fine = _blur(rng.standard_normal((R, R)), 3)
    return np.clip(0.5 + 0.18 * coarse / coarse.std() + 0.06 * fine / fine.std(), 0, 1)


def _noisy(x, seed, sigma=2 / 255):
    return np.clip(x + sigma * np.random.default_rng(seed).standard_normal(x.shape), 0, 1)


def _with_square(background, square, top, left):
    out = background.copy()
    out[top : top + square.shape[0], left : left + square.shape[1]] = square
    return out


def _patches_touching(top, left, size):
    covered = np.zeros((R, R), bool)
    covered[top : top + size, left : left + size] = True
    return covered.reshape(G, P, G, P).any((1, 3))


class TestPatchChange(unittest.TestCase):
    """The threshold these scenes allow is between 0.010 and 0.020; the test
    uses its two ends so a signal that loses one of its three steps fails."""

    UNCHANGED_BELOW = 0.010
    CHANGED_ABOVE = 0.020

    def setUp(self):
        self.bg = _texture(0)
        self.obj = _texture(1)[:64, :64]

    def test_identical_frames_are_zero(self):
        self.assertEqual(float(mc.patch_change(self.bg, self.bg).max()), 0.0)

    def test_sensor_noise_is_unchanged(self):
        d = mc.patch_change(_noisy(self.bg, 1), _noisy(self.bg, 2))
        self.assertLess(d.max(), self.UNCHANGED_BELOW)

    def test_brightness_change_is_unchanged(self):
        d = mc.patch_change(_noisy(self.bg, 1), _noisy(np.clip(self.bg * 1.05, 0, 1), 2))
        self.assertLess(d.max(), self.UNCHANGED_BELOW)

    def test_one_pixel_shake_is_unchanged(self):
        shaken = np.roll(self.bg, (1, 1), (0, 1))
        d = mc.patch_change(_noisy(self.bg, 1), _noisy(shaken, 2))
        inner = d[1:-1, 1:-1]  # np.roll wraps the border; a real shake does not
        self.assertLess(inner.max(), self.UNCHANGED_BELOW)

    def test_without_shift_tolerance_shake_reads_as_change(self):
        """The control for the test above: `tol` is what absorbs the shake."""
        shaken = np.roll(self.bg, (1, 1), (0, 1))
        d = mc.patch_change(_noisy(self.bg, 1), _noisy(shaken, 2), tol=0)
        self.assertGreater(d[1:-1, 1:-1].max(), self.CHANGED_ABOVE)

    def test_moving_textured_object_is_changed_and_only_there(self):
        a = _noisy(_with_square(self.bg, self.obj, 96, 96), 1)
        b = _noisy(_with_square(self.bg, self.obj, 96, 104), 2)  # half a patch to the right
        d = mc.patch_change(a, b)
        touched = _patches_touching(96, 96, 64) | _patches_touching(96, 104, 64)
        self.assertGreater(np.median(d[touched]), self.CHANGED_ABOVE)
        self.assertLess(d[~touched].max(), self.UNCHANGED_BELOW)

    def test_inside_of_a_flat_moving_object_is_unchanged(self):
        """Intended: where copying is correct, the patch must read as unchanged."""
        flat = np.full((64, 64), 0.9)
        a = _noisy(_with_square(self.bg, flat, 96, 64), 1)
        b = _noisy(_with_square(self.bg, flat, 96, 96), 2)  # the two positions overlap by half
        d = mc.patch_change(a, b)
        both = np.zeros((R, R), bool)
        both[96:160, 96:128] = True
        inside_both = both.reshape(G, P, G, P).all((1, 3))
        self.assertTrue(inside_both.any())
        self.assertLess(d[inside_both].max(), self.UNCHANGED_BELOW)

    def test_shifted_keeps_shape_and_moves_content(self):
        frame = np.arange(R * R, dtype=np.float64).reshape(R, R)
        moved = mc._shifted(frame, 2, -3)
        self.assertEqual(moved.shape, frame.shape)
        self.assertEqual(moved[10, 10], frame[8, 13])


class TestOnFakeDavis(unittest.TestCase):
    """Two 30-frame videos: one with a still camera and a moving square, one
    where the whole frame pans."""

    def _write(self, root, name, frames, masks):
        frame_dir = os.path.join(root, "JPEGImages", "480p", name)
        mask_dir = os.path.join(root, "Annotations", "480p", name)
        os.makedirs(frame_dir)
        os.makedirs(mask_dir)
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            Image.fromarray((255 * frame).astype(np.uint8)).convert("RGB").save(
                os.path.join(frame_dir, f"{i:05d}.jpg"), quality=95
            )
            Image.fromarray(mask.astype(np.uint8), mode="P").save(
                os.path.join(mask_dir, f"{i:05d}.png")
            )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        bg, obj = _texture(0), _texture(1)[:64, :64]
        still_frames, still_masks, pan_frames, pan_masks = [], [], [], []
        for i in range(30):
            left = 32 + 4 * i
            still_frames.append(_noisy(_with_square(bg, obj, 96, left), i))
            mask = np.zeros((R, R), np.uint8)
            mask[96:160, left : left + 64] = 1
            still_masks.append(mask)
            pan_frames.append(_noisy(np.roll(bg, 6 * i, 1), 100 + i))
            mask = np.zeros((R, R), np.uint8)
            mask[:16, :16] = 1
            pan_masks.append(mask)
        self._write(root, "still", still_frames, still_masks)
        self._write(root, "pan", pan_frames, pan_masks)
        os.makedirs(os.path.join(root, "ImageSets", "2017"))
        with open(os.path.join(root, "ImageSets", "2017", "val.txt"), "w") as f:
            f.write("still\npan\n")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def _measure(self, name, gaps=(6,)):
        from evals.video_tracking_frozen.davis import sequence_files

        return mc.measure_sequence(*sequence_files(self.root, name), gaps=list(gaps), tol=1)

    def test_still_camera_background_unchanged_object_changed(self):
        d = self._measure("still")[6]
        self.assertGreater((d["background"] < 0.012).mean(), 0.9)
        self.assertGreater((d["object"] > 0.012).mean(), 0.7)

    def test_panning_camera_changes_the_background(self):
        d = self._measure("pan")[6]
        self.assertGreater((d["background"] > 0.012).mean(), 0.7)

    def test_a_longer_gap_changes_more_of_the_object(self):
        """The square moves 4 px per frame: within tolerance-plus-noise at a
        gap of 1 frame for much of it, clearly moved at 6."""
        by_gap = self._measure("still", gaps=(1, 6))
        self.assertGreater(
            np.median(by_gap[6]["object"]), 1.5 * np.median(by_gap[1]["object"])
        )

    def test_shift_tolerance_reaches_the_signal(self):
        from evals.video_tracking_frozen.davis import sequence_files

        files = sequence_files(self.root, "still")
        strict = mc.measure_sequence(*files, gaps=[1], tol=0)[1]["object"]
        loose = mc.measure_sequence(*files, gaps=[1], tol=2)[1]["object"]
        self.assertTrue((loose <= strict + 1e-12).all())
        self.assertLess(loose.mean(), 0.9 * strict.mean())

    def test_gap_longer_than_the_video_is_left_out(self):
        self.assertEqual(self._measure("still", gaps=(6, 40)).keys(), {6})

    def test_main_prints_three_tables_and_writes_json(self):
        import contextlib
        import io
        import json

        out = os.path.join(self.root, "change.json")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = mc.main(["--davis", self.root, "--gaps", "1", "6", "--out", out])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        for marker in ("1. Share of patches", "2. Is there a gap", "3. Per video"):
            self.assertIn(marker, text)
        with open(out) as f:
            saved = json.load(f)
        self.assertEqual(set(saved["sequences"]), {"still", "pan"})
        self.assertGreater(saved["sequences"]["pan"]["background"], 0.7)
        self.assertLess(saved["sequences"]["still"]["background"], 0.1)


if __name__ == "__main__":
    unittest.main()
