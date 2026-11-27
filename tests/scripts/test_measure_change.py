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


class TestSharedShift(unittest.TestCase):
    def setUp(self):
        self.bg = _texture(0)
        self.obj = _texture(1)[:64, :64]

    def test_still_frames_give_zero(self):
        self.assertEqual(mc.shared_shift(_noisy(self.bg, 1), _noisy(self.bg, 2)), (0, 0))

    def test_frames_with_no_detail_give_zero(self):
        """Every shift fits a blank frame equally well; the answer must be 'no shift'."""
        blank = np.full((R, R), 0.5)
        self.assertEqual(mc.shared_shift(blank, blank), (0, 0))

    def test_a_pan_is_recovered_and_undone(self):
        panned = np.roll(self.bg, (8, -20), (0, 1))  # content moves 8 down, 20 left
        a, b = _noisy(self.bg, 1), _noisy(panned, 2)
        dy, dx = mc.shared_shift(a, b)
        self.assertEqual((dy, dx), (-8, 20))
        d = mc.patch_change(a, mc._shifted(b, dy, dx))
        self.assertLess(d[mc.covered_patches(dy, dx)].max(), 0.010)

    def test_a_pan_that_is_not_a_multiple_of_the_cell_is_recovered_to_the_pixel(self):
        """A leftover of 2 px would read as change everywhere (tolerance: 1 px)."""
        panned = np.roll(self.bg, (-3, 30), (0, 1))
        a, b = _noisy(self.bg, 1), _noisy(panned, 2)
        dy, dx = mc.shared_shift(a, b)
        self.assertEqual((dy, dx), (3, -30))
        d = mc.patch_change(a, mc._shifted(b, dy, dx))
        self.assertLess(d[mc.covered_patches(dy, dx)].max(), 0.010)

    def test_a_moving_object_does_not_decide_the_shift(self):
        a = _noisy(_with_square(self.bg, self.obj, 96, 64), 1)
        b = _noisy(_with_square(self.bg, self.obj, 96, 104), 2)
        self.assertEqual(mc.shared_shift(a, b), (0, 0))

    def test_a_large_vivid_object_held_in_frame_does_not_decide_the_shift(self):
        """A tracking shot: the camera follows the object, so the object stays
        put in the image and the pale background slides past.  The background
        is the larger part and must decide.  (An average would follow the
        vivid object; the median follows the majority.)"""
        pale = 0.5 + 0.3 * (self.bg - 0.5)
        vivid = np.clip(0.5 + 2.5 * (_texture(2)[:152, :152] - 0.5), 0, 1)  # 35% of the frame
        a = _noisy(_with_square(pale, vivid, 52, 52), 1)
        b = _noisy(_with_square(np.roll(pale, 12, 1), vivid, 52, 52), 2)
        self.assertEqual(mc.shared_shift(a, b), (0, -12))

    def test_covered_patches_drops_the_uncovered_border(self):
        covered = mc.covered_patches(-8, 20)
        self.assertFalse(covered[-1].any())  # moved up: bottom row is repeated edge
        self.assertFalse(covered[:, :2].any())  # moved right by 20 px: two columns
        self.assertTrue(covered[:-1, 2:].all())
        self.assertTrue(mc.covered_patches(0, 0).all())


class TestWeightAndRegime(unittest.TestCase):
    def test_weight_rises_in_proportion_and_stops_at_one(self):
        weights = mc.unexplained_weight([0.0, 0.008, 0.016, 0.032, 0.5], full=0.032)
        np.testing.assert_allclose(weights, [0.0, 0.25, 0.5, 1.0, 1.0])

    def test_regime_names_the_four_cases(self):
        self.assertEqual(mc.regime(0.1, 0.1), mc.REGIMES[0])  # little changes
        self.assertEqual(mc.regime(0.9, 0.1), mc.REGIMES[1])  # shared motion explains it
        self.assertEqual(mc.regime(0.9, 0.5), mc.REGIMES[2])
        self.assertEqual(mc.regime(0.9, 0.9), mc.REGIMES[3])
        self.assertEqual(len(set(mc.REGIMES)), 4)


    def test_weight_table_uses_the_change_left_after_undoing(self):
        """An object the shared motion fully explains must get none of the
        weight afterwards, however much it changed as measured."""
        one_video = {
            6: {
                "background": np.full(90, 0.05),
                "object": np.full(10, 0.05),
                "background_shared": np.full(90, 0.02),
                "object_shared": np.zeros(10),
                "shift": np.array([8.0]),
            }
        }
        result = mc.report(
            {"v": one_video}, [6], mc.DAVIS_KINDS, lambda gap: gap / 24, print_fn=lambda _: None
        )
        row = result["weight_on_objects"][mc.WEIGHT_FULL[0]]
        self.assertAlmostEqual(row["raw"], 0.1)
        self.assertEqual(row["_shared"], 0.0)


class TestOnFakeDavis(unittest.TestCase):
    """Three 30-frame videos: a still camera with a moving square, a pan with
    nothing else moving, and a pan with a square moving on its own."""

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
        both_frames, both_masks = [], []
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
            top = 40 + 4 * i  # the square moves down while the camera pans sideways
            both_frames.append(_noisy(_with_square(np.roll(bg, 6 * i, 1), obj, top, 96), 200 + i))
            mask = np.zeros((R, R), np.uint8)
            mask[top : top + 64, 96:160] = 1
            both_masks.append(mask)
        self._write(root, "still", still_frames, still_masks)
        self._write(root, "pan", pan_frames, pan_masks)
        self._write(root, "pan-object", both_frames, both_masks)
        os.makedirs(os.path.join(root, "ImageSets", "2017"))
        with open(os.path.join(root, "ImageSets", "2017", "val.txt"), "w") as f:
            f.write("still\npan\npan-object\n")
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

    def test_undoing_the_camera_clears_a_pure_pan(self):
        d = self._measure("pan")[6]
        self.assertEqual(float(np.median(d["shift"])), 36.0)  # 6 px a frame, 6 frames
        self.assertGreater((d["background"] > 0.012).mean(), 0.7)
        self.assertLess((d["background_shared"] > 0.012).mean(), 0.1)

    def test_undoing_the_camera_keeps_what_moves_on_its_own(self):
        d = self._measure("pan-object")[6]
        # Strict on purpose: the object's patches count as background, and
        # show up here, if its mask is not moved together with its frame.
        self.assertLess((d["background_shared"] > 0.012).mean(), 0.01)
        self.assertGreater((d["object_shared"] > 0.012).mean(), 0.9)

    def test_undoing_the_camera_changes_nothing_when_it_is_still(self):
        d = self._measure("still")[6]
        self.assertEqual(float(np.max(d["shift"])), 0.0)
        np.testing.assert_array_equal(d["background"], d["background_shared"])

    def test_gap_longer_than_the_video_is_left_out(self):
        self.assertEqual(self._measure("still", gaps=(6, 40)).keys(), {6})

    def test_main_prints_four_tables_and_writes_json(self):
        import contextlib
        import io
        import json

        out = os.path.join(self.root, "change.json")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = mc.main(["--davis", self.root, "--gaps", "1", "6", "--out", out])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        for marker in (
            "1. Share of patches",
            "2. Is there a gap",
            "3. Per video",
            "4. All videos",
            "5. What kind of video",
            "6. Does the weight land on objects",
        ):
            self.assertIn(marker, text)
        with open(out) as f:
            saved = json.load(f)
        self.assertEqual(set(saved["videos"]), {"still", "pan", "pan-object"})
        self.assertLess(saved["videos"]["pan"]["background_shared"], 0.1)
        # one still video, two that the shared motion explains
        self.assertEqual(saved["regimes"], dict(zip(mc.REGIMES, [1, 2, 0, 0])))
        # undoing the shared motion must move the weight onto the object
        on_objects = saved["weight_on_objects"]
        for full in mc.WEIGHT_FULL:
            row = on_objects[str(full)]
            self.assertLess(row["raw"], 2 * on_objects["by_count"])
            self.assertGreater(row["_shared"], 4 * on_objects["by_count"])
        self.assertGreater(saved["videos"]["pan"]["background"], 0.7)
        self.assertLess(saved["videos"]["still"]["background"], 0.1)


class TestOnTrainingVideos(unittest.TestCase):
    """`--videos`, with a stand-in for the video reader: 60 frames at 30 fps,
    the camera panning 2 px a frame and a square moving down on its own."""

    def _reader(self, path):
        if "broken" in path:
            raise OSError("cannot open")
        bg, obj = _texture(0), _texture(1)[:64, :64]

        def read(indices):
            frames = [
                _noisy(_with_square(np.roll(bg, 2 * i, 1), obj, 40 + 2 * i, 96), i)
                for i in indices
            ]
            return [np.repeat((255 * f).astype(np.uint8)[:, :, None], 3, 2) for f in frames]

        return 60, 30.0, read

    def _measure(self, gaps, pairs=3):
        rng = np.random.default_rng(0)
        return mc.measure_video("clip.mp4", gaps, 1, pairs, rng, open_video=self._reader)

    def test_gaps_are_in_seconds_and_follow_the_frame_rate(self):
        found = self._measure([0.5])
        self.assertEqual(found[0.5]["shift"].tolist(), [30.0] * 3)  # 15 frames x 2 px

    def test_a_gap_below_one_frame_uses_one_frame_and_a_long_one_is_left_out(self):
        found = self._measure([0.01, 5.0])
        self.assertEqual(set(found), {0.01})
        self.assertEqual(found[0.01]["shift"].tolist(), [2.0] * 3)  # one frame: 2 px

    def test_all_patches_are_counted_and_the_camera_is_undone(self):
        found = self._measure([0.5])[0.5]
        self.assertEqual(len(found["all"]), 3 * G * G)
        self.assertGreater((found["all"] > 0.012).mean(), 0.7)
        # after undoing the pan only the square's old and new places differ
        self.assertLess((found["all_shared"] > 0.012).mean(), 0.25)
        self.assertGreater((found["all_shared"] > 0.012).mean(), 0.03)

    def test_read_video_list_takes_the_path_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.csv")
            with open(path, "w") as f:
                f.write("/data/a.mp4 0\n/data/with space/b.mp4 0\n\n/data/c.mp4\n")
            self.assertEqual(
                mc.read_video_list(path), ["/data/a.mp4", "/data/with space/b.mp4", "/data/c.mp4"]
            )

    def test_main_samples_videos_skips_a_broken_one_and_shows_the_spread(self):
        import contextlib
        import io
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.csv")
            with open(path, "w") as f:
                f.write("".join(f"/data/v{i}.mp4 0\n" for i in range(5)) + "/data/broken.mp4 0\n")
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(mc, "_open_with_decord", self._reader):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = mc.main(["--videos", path, "--n-videos", "6", "--pairs", "2"])
        self.assertEqual(code, 0)
        self.assertIn("5 of 6 training videos (1 unreadable)", out.getvalue())
        self.assertIn("50% of videos below", out.getvalue())
        self.assertIn("5. What kind of video", out.getvalue())
        self.assertNotIn("6. Does the weight", out.getvalue())  # needs masks
        self.assertIn("broken.mp4", err.getvalue())


if __name__ == "__main__":
    unittest.main()
