# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""app/vjepa_3/masks.py: random truncation and whole-time-step masks.

Sizes are the proxy recipe's: 16 frames -> 8 time-steps of 16 x 16 tokens.
"""

import unittest
from pathlib import Path

import torch
import yaml

from app.vjepa_3.masks import (
    MaskCollatorV3,
    _MaskGeneratorV3,
    build_mask_collator,
    uses_v3_masking,
)
from src.masks.multiseq_multiblock3d import MaskCollator, _MaskGenerator

REPO_ROOT = Path(__file__).resolve().parents[2]
LADDER = REPO_ROOT / "configs" / "train_3" / "ladder"
STEPS, PER_STEP = 8, 256
SMALL_BLOCKS = dict(
    crop_size=(256, 256),
    num_frames=16,
    spatial_patch_size=(16, 16),
    temporal_patch_size=2,
    spatial_pred_mask_scale=[0.15, 0.15],
    temporal_pred_mask_scale=[1.0, 1.0],
    aspect_ratio=[0.75, 1.5],
    npred=8,
)


def _collator(arm):
    cfg = yaml.safe_load((LADDER / f"{arm}.yaml").read_text())
    return build_mask_collator(
        cfgs_mask=cfg["mask"],
        dataset_fpcs=cfg["data"]["dataset_fpcs"],
        crop_size=cfg["data"]["crop_size"],
        patch_size=cfg["data"]["patch_size"],
        tubelet_size=cfg["data"]["tubelet_size"],
    )


def _share_per_step(ids):
    """Share of each time-step's tokens that appear in `ids` ([B, n])."""
    counts = torch.bincount((ids // PER_STEP).flatten(), minlength=STEPS).float()
    return counts / (ids.size(0) * PER_STEP)


class TestDefaultIsUpstream(unittest.TestCase):
    def test_default_configs_get_the_2_1_collator_itself(self):
        for arm in ("a0-control", "s1-sigreg", "t1-future", "t2-fps12", "w0-warm-control"):
            with self.subTest(arm=arm):
                self.assertIs(type(_collator(arm)), MaskCollator)
        for arm in ("t3-frames", "t4-whole-clip", "w5-warm-frames", "w6-warm-whole-clip"):
            with self.subTest(arm=arm):
                self.assertIs(type(_collator(arm)), MaskCollatorV3)

    def test_v3_generator_at_defaults_reproduces_2_1_exactly(self):
        """Same arguments, same RNG state: identical masks, call after call.
        This is what makes `truncation` and `hide_time_steps` the ONLY things
        an arm changes when it switches to the v3 generator."""
        for extra in ({}, {"max_context_frames_ratio": 0.5}, {"max_keep": 300}):
            ours = _MaskGeneratorV3(**SMALL_BLOCKS, **extra)
            theirs = _MaskGenerator(**SMALL_BLOCKS, **extra)
            for call in range(3):
                torch.manual_seed(call)
                a = ours(batch_size=8)
                torch.manual_seed(call)
                b = theirs(batch_size=8)
                with self.subTest(extra=extra, call=call):
                    self.assertTrue(torch.equal(a[0], b[0]))
                    self.assertTrue(torch.equal(a[1], b[1]))

    def test_detection_of_the_two_options(self):
        base = {"num_blocks": 8}
        self.assertFalse(uses_v3_masking([base, dict(base, truncation="tail", hide_time_steps=0)]))
        self.assertTrue(uses_v3_masking([base, dict(base, truncation="random")]))
        self.assertTrue(uses_v3_masking([dict(base, hide_time_steps=2), base]))


class TestRandomTruncation(unittest.TestCase):
    def _pair(self, seed=0, batch=16):
        tail = _MaskGeneratorV3(**SMALL_BLOCKS)
        rand = _MaskGeneratorV3(**SMALL_BLOCKS, truncation="random")
        torch.manual_seed(seed)
        t = tail(batch_size=batch)
        torch.manual_seed(seed)
        r = rand(batch_size=batch)
        return t, r

    def test_same_budget_sorted_disjoint_and_inside_the_same_blocks(self):
        (t_enc, t_pred), (r_enc, r_pred) = self._pair()
        self.assertEqual(t_enc.shape, r_enc.shape)
        self.assertEqual(t_pred.shape, r_pred.shape)
        for ids in (r_enc, r_pred):
            self.assertTrue((ids[:, 1:] > ids[:, :-1]).all(), "ids must stay sorted and unique")
        for e, p in zip(r_enc, r_pred):
            self.assertEqual(len(set(e.tolist()) & set(p.tolist())), 0)
        # Blocks hide the same (row, column) in every frame.  A token kept as
        # context must never sit at a hidden position, whatever the cut.
        for e, p in zip(r_enc, r_pred):
            self.assertEqual(len(set((e % PER_STEP).tolist()) & set((p % PER_STEP).tolist())), 0)

    def test_the_end_of_the_clip_is_used(self):
        """The measured gap this option exists for: with the 2.1 cut the last
        two time-steps are nearly empty; with a random cut every time-step
        gets the same share."""
        tail_enc, tail_pred, rand_enc, rand_pred = [], [], [], []
        for seed in range(20):
            (te, tp), (re_, rp) = self._pair(seed)
            tail_enc.append(_share_per_step(te))
            tail_pred.append(_share_per_step(tp))
            rand_enc.append(_share_per_step(re_))
            rand_pred.append(_share_per_step(rp))
        tail_enc, tail_pred = torch.stack(tail_enc).mean(0), torch.stack(tail_pred).mean(0)
        rand_enc, rand_pred = torch.stack(rand_enc).mean(0), torch.stack(rand_pred).mean(0)
        self.assertLess(float(tail_enc[-1]), 0.10)
        self.assertLess(float(tail_pred[-1]), 0.20)
        for share in (rand_enc, rand_pred):
            self.assertLess(float(share.max() - share.min()), 0.03)
        self.assertAlmostEqual(float(rand_enc.sum()), float(tail_enc.sum()), places=5)

    def test_unknown_value_raises(self):
        with self.assertRaisesRegex(ValueError, "truncation must be one of"):
            _MaskGeneratorV3(**SMALL_BLOCKS, truncation="head")


class TestWholeTimeSteps(unittest.TestCase):
    def test_exactly_k_steps_are_hidden_and_everything_else_is_shown(self):
        for k in (1, 3, 6):
            gen = _MaskGeneratorV3(**SMALL_BLOCKS, hide_time_steps=k)
            torch.manual_seed(k)
            enc, pred = gen(batch_size=8)
            with self.subTest(k=k):
                self.assertEqual(tuple(enc.shape), (8, (STEPS - k) * PER_STEP))
                self.assertEqual(tuple(pred.shape), (8, k * PER_STEP))
                patterns = set()
                for e, p in zip(enc, pred):
                    hidden = sorted(set((p // PER_STEP).tolist()))
                    shown = sorted(set((e // PER_STEP).tolist()))
                    self.assertEqual(len(hidden), k)
                    self.assertEqual(sorted(hidden + shown), list(range(STEPS)))
                    self.assertEqual(len(set(e.tolist()) | set(p.tolist())), STEPS * PER_STEP)
                    patterns.add(tuple(hidden))
                self.assertGreater(len(patterns), 1, "every clip hid the same moments")

    def test_every_time_step_is_hidden_about_equally_often(self):
        gen = _MaskGeneratorV3(**SMALL_BLOCKS, hide_time_steps=6)
        torch.manual_seed(0)
        share = torch.stack([_share_per_step(gen(batch_size=16)[1]) for _ in range(40)]).mean(0)
        self.assertLess(float((share - 0.75).abs().max()), 0.06)

    def test_bad_values_raise(self):
        for k in (-1, STEPS):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "hide_time_steps must be"):
                _MaskGeneratorV3(**SMALL_BLOCKS, hide_time_steps=k)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _MaskGeneratorV3(**SMALL_BLOCKS, hide_time_steps=2, full_complement=True)

    def test_the_arm_matches_the_controls_visible_budget(self):
        """t3-frames was sized so mask 0 shows about what the control's shows."""
        torch.manual_seed(0)
        frames_enc, _ = _collator("t3-frames").mask_generators[16][0](batch_size=16)
        sizes = []
        control = _collator("a0-control").mask_generators[16][0]
        for _ in range(20):
            sizes.append(control(batch_size=16)[0].shape[1])
        self.assertEqual(frames_enc.shape[1], 512)
        self.assertLess(abs(sum(sizes) / len(sizes) - 512), 60)
        # the other mask of the arm is the control's, from the same code path
        other = _collator("t3-frames").mask_generators[16][1]
        self.assertEqual((other.hide_time_steps, other.truncation), (0, "tail"))


if __name__ == "__main__":
    unittest.main()
