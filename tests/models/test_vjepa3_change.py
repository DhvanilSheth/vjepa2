# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""app/vjepa_3/change.py: from a clip and a mask to one weight per hidden
token, and the weighted loss term.

The signal itself (patch_change, shared_shift, ...) is tested in
tests/scripts/test_measure_change.py, on scenes with a known answer.

THE SCENE used here: 8 time-steps of 256 x 256.  The camera pans 6 pixels a
time-step; a 64-pixel square moves down 8 pixels a time-step on its own.
Time-steps 1 and 6 are shown; the rest are asked for.
"""

import unittest

import numpy as np
import torch

from app.vjepa_3 import change
from app.vjepa_3.masks import MaskCollatorV3, build_mask_collator
from src.masks.multiseq_multiblock3d import MaskCollator
from tests.scripts.test_measure_change import _noisy, _texture, _with_square

SIZE, PATCH, TUBELET, STEPS = 256, 16, 2, 8
GRID = SIZE // PATCH
PER_STEP = GRID * GRID
PAN, DROP = 6, 8
SHOWN = [1, 6]


def _gray_frame(step, moving=True, seed=None):
    background, square = _texture(0), _texture(1)[:64, :64]
    if not moving:
        return _noisy(_with_square(background, square, 40, 96), step if seed is None else seed)
    panned = np.roll(background, PAN * step, 1)
    return _noisy(_with_square(panned, square, 40 + DROP * step, 96), step)


def _clip(frames):
    """[T', H, W] grayscale in [0, 1] -> [3, 2 * T', H, W] as the model gets it."""
    gray = torch.from_numpy(np.stack(frames)).float().repeat_interleave(TUBELET, dim=0)
    mean = torch.tensor(change.IMAGENET_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(change.IMAGENET_STD).view(3, 1, 1, 1)
    return (gray[None].expand(3, -1, -1, -1) - mean) / std


def _ids(steps):
    return np.concatenate([np.arange(PER_STEP) + s * PER_STEP for s in steps])


def _object_patches(step):
    covered = np.zeros((SIZE, SIZE), bool)
    covered[40 + DROP * step : 104 + DROP * step, 96:160] = True
    share = covered.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3))
    return share >= 0.25, share == 0


class TestGrayTimeSteps(unittest.TestCase):
    def test_it_undoes_the_normalisation_and_takes_one_frame_per_time_step(self):
        frames = [_gray_frame(s) for s in range(STEPS)]
        clip = _clip(frames)
        clip[:, 1::TUBELET] = 0.0  # the second frame of every time-step must not be read
        out = change.gray_time_steps(clip, TUBELET)
        self.assertEqual(out.shape, (STEPS, SIZE, SIZE))
        np.testing.assert_allclose(out, np.stack(frames), atol=1e-5)


class TestShownTimeSteps(unittest.TestCase):
    def test_whole_time_steps_are_listed(self):
        self.assertEqual(change.shown_time_steps(_ids([6, 1]), PER_STEP), [1, 6])

    def test_a_partly_visible_time_step_is_refused(self):
        with self.assertRaisesRegex(ValueError, "whole time-steps"):
            change.shown_time_steps(_ids([1])[:-1], PER_STEP)


class TestTargetWeights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clip = _clip([_gray_frame(s) for s in range(STEPS)])
        cls.asked_steps = [s for s in range(STEPS) if s not in SHOWN]
        cls.weights = change.target_weights(
            cls.clip, _ids(SHOWN), _ids(cls.asked_steps), TUBELET, PATCH
        ).reshape(len(cls.asked_steps), GRID, GRID)

    def grid(self, step):
        return self.weights[self.asked_steps.index(step)]

    def test_one_weight_per_asked_token_between_0_and_1(self):
        self.assertEqual(self.weights.size, len(self.asked_steps) * PER_STEP)
        self.assertGreaterEqual(self.weights.min(), 0.0)
        self.assertLessEqual(self.weights.max(), 1.0)

    def test_the_object_counts_and_the_background_does_not(self):
        for step in (2, 3, 4, 5):  # between the two shown time-steps
            on_object, on_background = _object_patches(step)
            with self.subTest(step=step):
                self.assertGreater(self.grid(step)[on_object].mean(), 0.85)
                self.assertLess(self.grid(step)[on_background].mean(), 0.1)

    def test_what_was_never_in_view_counts_fully(self):
        """Step 0 comes before both shown time-steps.  The camera pans one
        way, so the strip at one edge of step 0 is in neither of them: it
        cannot be copied from anywhere."""
        grid = self.grid(0)
        self.assertEqual(float(grid[:, -1].min()), 1.0)
        _, on_background = _object_patches(0)
        on_background[:, -1] = False
        self.assertLess(grid[on_background].mean(), 0.1)

    def test_the_object_gets_most_of_the_weight(self):
        """16 of 256 patches (6%) hold about two thirds of the weight."""
        on_object, _ = _object_patches(3)
        grid = self.grid(3)
        self.assertEqual(int(on_object.sum()), 16)
        self.assertGreater(grid[on_object].sum() / grid.sum(), 0.6)

    def test_one_shown_time_step_that_explains_a_patch_is_enough(self):
        """Background the square covered at step 1 (row 3, column 8 at step 3)
        and the left edge, which was out of view at step 1, cannot be copied
        from step 1.  They can from step 6, so with both shown they count for
        little."""
        only_first = change.target_weights(
            self.clip, _ids([1]), _ids([3]), TUBELET, PATCH
        ).reshape(GRID, GRID)
        for row, column in ((3, 8), (8, 0)):
            with self.subTest(patch=(row, column)):
                self.assertGreater(only_first[row, column], 0.9)
                self.assertLess(self.grid(3)[row, column], 0.1)

    def test_weights_follow_the_order_of_the_asked_ids(self):
        ids = _ids([3])
        order = np.random.default_rng(0).permutation(len(ids))
        shuffled = change.target_weights(self.clip, _ids(SHOWN), ids[order], TUBELET, PATCH)
        np.testing.assert_array_equal(shuffled, self.grid(3).ravel()[order])

    def test_a_still_clip_has_almost_nothing_to_ask_for(self):
        clip = _clip([_gray_frame(s, moving=False) for s in range(STEPS)])
        weights = change.target_weights(clip, _ids(SHOWN), _ids([3]), TUBELET, PATCH)
        self.assertLess(weights.max(), 0.15)

    def test_a_larger_scale_gives_smaller_weights(self):
        ids = _ids([3])
        low = change.target_weights(self.clip, _ids(SHOWN), ids, TUBELET, PATCH, full=0.032)
        high = change.target_weights(self.clip, _ids(SHOWN), ids, TUBELET, PATCH, full=0.128)
        self.assertTrue((high <= low + 1e-7).all())
        self.assertLess(high.sum(), 0.8 * low.sum())


class TestResolveTargetWeights(unittest.TestCase):
    def test_absent_or_none_means_no_weights(self):
        self.assertEqual(change.resolve_target_weights([{}, {"target_weight": "none"}]), [None, None])
        self.assertEqual(change.resolve_target_weights([{"target_weight": None}]), [None])

    def test_unexplained_gives_the_scale(self):
        masks = [{"hide_time_steps": 6, "target_weight": "unexplained"}, {}]
        self.assertEqual(change.resolve_target_weights(masks), [{"full": 0.064}, None])
        masks[0]["target_weight_full"] = 0.03
        self.assertEqual(change.resolve_target_weights(masks)[0], {"full": 0.03})

    def test_settings_that_cannot_work_are_refused(self):
        on = {"hide_time_steps": 6, "target_weight": "unexplained"}
        for masks, message in (
            ([{"target_weight": "unexplained"}], "hide_time_steps"),
            ([{"hide_time_steps": 6, "target_weight": "motion"}], "must be one of"),
            ([dict(on, target_weight_full=0)], "positive"),
        ):
            with self.subTest(masks=masks), self.assertRaisesRegex(ValueError, message):
                change.resolve_target_weights(masks)
        with self.assertRaisesRegex(ValueError, "class token"):
            change.resolve_target_weights([on], has_cls_first=True)


class TestWeightedError(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(0)
        self.prediction = torch.randn(3, 10, 8, generator=gen)
        self.target = torch.randn(3, 10, 8, generator=gen)
        self.weights = torch.rand(3, 10, generator=gen)

    def test_all_ones_is_the_plain_loss(self):
        for exponent in (1.0, 2.0):
            plain = torch.mean(torch.abs(self.prediction - self.target) ** exponent) / exponent
            ours = change.weighted_error(
                self.prediction, self.target, torch.ones(3, 10), exponent
            )
            self.assertAlmostEqual(float(ours), float(plain), places=6)

    def test_it_is_the_weighted_average_of_the_token_errors(self):
        per_token = torch.abs(self.prediction - self.target).mean(-1)
        expected = (per_token * self.weights).sum() / self.weights.sum()
        ours = change.weighted_error(self.prediction, self.target, self.weights, 1.0)
        self.assertAlmostEqual(float(ours), float(expected), places=6)

    def test_a_token_with_weight_zero_does_not_matter(self):
        self.weights[0, 0] = 0.0
        before = change.weighted_error(self.prediction, self.target, self.weights, 1.0)
        self.target[0, 0] += 100.0
        after = change.weighted_error(self.prediction, self.target, self.weights, 1.0)
        self.assertEqual(float(before), float(after))

    def test_each_token_pushes_in_proportion_to_its_weight(self):
        """With exponent 1 the slope is +1 or -1 whatever the error, so the
        size of a token's gradient is its weight divided by the total."""
        prediction = self.prediction.clone().requires_grad_(True)
        change.weighted_error(prediction, self.target, self.weights, 1.0).backward()
        push = prediction.grad.abs().sum(-1)
        expected = self.weights / self.weights.sum()
        torch.testing.assert_close(push, expected, rtol=1e-5, atol=1e-8)

    def test_nothing_to_ask_for_gives_zero_not_a_division_by_zero(self):
        loss = change.weighted_error(self.prediction, self.target, torch.zeros(3, 10), 1.0)
        self.assertEqual(float(loss), 0.0)

    def test_tiny_weights_do_not_blow_the_loss_up(self):
        tiny = torch.full((3, 10), 1e-4)
        plain = torch.mean(torch.abs(self.prediction - self.target))
        self.assertLess(float(change.weighted_error(self.prediction, self.target, tiny, 1.0)), float(plain))

    def test_float16_inputs_are_summed_in_float32(self):
        """A float16 sum over a real batch (16 clips x 1,536 tokens x 3,072
        numbers of about 0.5) passes float16's largest value, 65,504."""
        prediction = torch.full((16, 1536, 64), 0.5, dtype=torch.float16)
        target = torch.zeros_like(prediction)
        loss = change.weighted_error(prediction, target, torch.ones(16, 1536, dtype=torch.float16), 1.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss), 0.5, places=4)


class TestCollatorSendsWeights(unittest.TestCase):
    """Small sizes: 8 frames of 64 x 64 -> 4 time-steps of 16 tokens."""

    MASKS = [
        dict(hide_time_steps=2, spatial_scale=[0.15, 0.15], temporal_scale=[1.0, 1.0],
             aspect_ratio=[0.75, 1.5], num_blocks=8),
        dict(spatial_scale=[0.7, 0.7], temporal_scale=[1.0, 1.0], aspect_ratio=[0.75, 1.5],
             num_blocks=2),
    ]

    def _collator(self, **first_mask):
        masks = [dict(self.MASKS[0], **first_mask), dict(self.MASKS[1])]
        return build_mask_collator(
            cfgs_mask=masks, dataset_fpcs=[8], crop_size=64, patch_size=16, tubelet_size=2
        )

    def _batch(self, n=3):
        gen = torch.Generator().manual_seed(0)
        return [
            ([torch.randn(3, 8, 64, 64, generator=gen)], 0, [np.arange(8)]) for _ in range(n)
        ]

    def test_without_the_key_the_batch_has_three_items(self):
        collator = self._collator()
        self.assertIsInstance(collator, MaskCollatorV3)
        (group,) = collator(self._batch())
        self.assertEqual(len(group), 3)

    def test_with_the_key_a_fourth_item_carries_the_weights(self):
        collator = self._collator(target_weight="unexplained")
        (group,) = collator(self._batch())
        collated, masks_enc, masks_pred, weights = group
        self.assertEqual(len(weights), 2)
        self.assertEqual(weights[0].shape, masks_pred[0].shape)
        self.assertEqual(weights[0].dtype, torch.float32)
        self.assertEqual(weights[1].numel(), 0)  # the second mask has no weights
        clips = collated[0][0]
        for b in range(clips.shape[0]):
            expected = change.target_weights(
                clips[b], masks_enc[0][b].numpy(), masks_pred[0][b].numpy(), 2, 16
            )
            np.testing.assert_array_equal(weights[0][b].numpy(), expected)

    def test_the_scale_in_the_config_reaches_the_weights(self):
        batch = self._batch()
        torch.manual_seed(0)
        (default,) = self._collator(target_weight="unexplained")(batch)
        torch.manual_seed(0)
        (wide,) = self._collator(target_weight="unexplained", target_weight_full=10.0)(batch)
        torch.testing.assert_close(wide[2][0], default[2][0])  # same masks
        self.assertTrue((wide[3][0] <= default[3][0]).all())
        self.assertLess(float(wide[3][0].mean()), 0.9 * float(default[3][0].mean()))

    def test_the_key_alone_switches_to_the_v3_collator_and_is_checked(self):
        with self.assertRaisesRegex(ValueError, "hide_time_steps"):
            build_mask_collator(
                cfgs_mask=[dict(self.MASKS[1], target_weight="unexplained")],
                dataset_fpcs=[8], crop_size=64, patch_size=16, tubelet_size=2,
            )
        plain = build_mask_collator(
            cfgs_mask=[dict(self.MASKS[1])], dataset_fpcs=[8], crop_size=64, patch_size=16,
            tubelet_size=2,
        )
        self.assertIs(type(plain), MaskCollator)


if __name__ == "__main__":
    unittest.main()
