# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Gates for the frozen-feature tracking probe (evals/video_tracking_frozen).

Everything runs on CPU with synthetic data.  The real DAVIS set, a trained
checkpoint, and agreement with the official DAVIS toolkit are NOT covered.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from evals.feature_clustering_frozen.eval import build_encoder
from evals.video_tracking_frozen import eval as tracking
from evals.video_tracking_frozen.davis import (
    load_frames,
    model_input_size,
    read_annotation,
    read_sequences,
    sequence_files,
)
from evals.video_tracking_frozen.metrics import (
    _dilate_disk,
    boundary_f_measure,
    region_similarity,
    seg2bmap,
)
from evals.video_tracking_frozen.propagate import (
    downsample_labels,
    labels_to_mask,
    neighborhood_mask,
    propagate_video,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROXY_BASE = REPO_ROOT / "configs" / "train_3" / "proxy" / "base-1gpu.yaml"
EVAL_CONFIG = REPO_ROOT / "configs" / "eval_3" / "vitb16" / "davis-tracking.yaml"


def _square(h, w, top, left, side):
    mask = np.zeros((h, w), dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


class TestMetrics(unittest.TestCase):
    def test_region_similarity_closed_forms(self):
        gt = _square(20, 20, 4, 4, 4)
        self.assertEqual(region_similarity(gt, gt), 1.0)
        self.assertEqual(region_similarity(_square(20, 20, 12, 12, 4), gt), 0.0)
        half = gt.copy()
        half[:, 6:] = False  # keeps 8 of the 16 pixels, adds none
        self.assertAlmostEqual(region_similarity(half, gt), 0.5)
        empty = np.zeros((20, 20), dtype=bool)
        self.assertEqual(region_similarity(empty, empty), 1.0)
        self.assertEqual(region_similarity(empty, gt), 0.0)

    def test_boundary_map(self):
        self.assertEqual(seg2bmap(np.zeros((9, 9), dtype=bool)).sum(), 0)
        self.assertEqual(seg2bmap(np.ones((9, 9), dtype=bool)).sum(), 0)
        boundary = seg2bmap(_square(20, 20, 5, 5, 6))
        self.assertGreater(boundary.sum(), 0)
        # nothing deep inside the square or far outside it is boundary
        self.assertFalse(boundary[7:9, 7:9].any())
        self.assertFalse(boundary[15:, 15:].any())

    def test_disk_dilation(self):
        point = np.zeros((11, 11), dtype=bool)
        point[5, 5] = True
        self.assertEqual(_dilate_disk(point, 2).sum(), 13)  # x^2 + y^2 <= 4
        self.assertEqual(_dilate_disk(point, 0).sum(), 1)

    def test_boundary_f_closed_forms(self):
        gt = _square(100, 100, 30, 30, 30)
        empty = np.zeros_like(gt)
        self.assertEqual(boundary_f_measure(gt, gt), 1.0)
        # 100x100 -> tolerance ceil(0.008 * 141.4) = 2 px
        self.assertEqual(boundary_f_measure(_square(100, 100, 31, 31, 30), gt), 1.0)
        self.assertEqual(boundary_f_measure(_square(100, 100, 2, 2, 10), gt), 0.0)
        self.assertEqual(boundary_f_measure(empty, gt), 0.0)
        self.assertEqual(boundary_f_measure(gt, empty), 0.0)
        self.assertEqual(boundary_f_measure(empty, empty), 1.0)

    def test_boundary_f_falls_with_displacement(self):
        gt = _square(100, 100, 30, 30, 30)
        scores = [boundary_f_measure(_square(100, 100, 30, 30 + d, 30), gt) for d in (0, 4, 12)]
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])


class TestPropagation(unittest.TestCase):
    def test_neighborhood_mask(self):
        centre = 2 * 5 + 2
        self.assertEqual(int(neighborhood_mask(5, 5, 1, "circle")[centre].sum()), 5)
        self.assertEqual(int(neighborhood_mask(5, 5, 1, "square")[centre].sum()), 9)
        self.assertEqual(int(neighborhood_mask(5, 5, 1, "square")[0].sum()), 4)  # corner
        self.assertIsNone(neighborhood_mask(5, 5, None))
        self.assertIsNone(neighborhood_mask(5, 5, 0))
        with self.assertRaises(ValueError):
            neighborhood_mask(5, 5, 1, "hexagon")

    def test_downsampling_keeps_sub_patch_objects(self):
        onehot = torch.zeros(2, 32, 32)
        onehot[1, 2:6, 2:6] = 1  # a 4x4-px object inside one 16-px patch
        onehot[0] = 1 - onehot[1]
        small = downsample_labels(onehot, 16)
        self.assertEqual(tuple(small.shape), (2, 2, 2))
        self.assertAlmostEqual(float(small[1, 0, 0]), 16 / 256)
        self.assertEqual(float(small[1].sum() - small[1, 0, 0]), 0.0)
        self.assertTrue(torch.allclose(small.sum(0), torch.ones(2, 2)))

    @staticmethod
    def _moving_square(n_frames=8, h=12, w=16, side=3, dim=8, noise=0.05, seed=0):
        """An object that moves one cell per frame over a distinct background."""
        gen = torch.Generator().manual_seed(seed)
        obj, bg = torch.randn(dim, generator=gen), torch.randn(dim, generator=gen)
        truth = torch.zeros(n_frames, h, w, dtype=torch.long)
        for t in range(n_frames):
            truth[t, 4 : 4 + side, 2 + t : 2 + t + side] = 1
        feats = torch.where(truth[..., None] == 1, obj, bg)
        feats = feats + noise * torch.randn(feats.shape, generator=gen)
        first = F.one_hot(truth[0], 2).permute(2, 0, 1).float()
        return feats, first, truth

    def test_discriminative_features_track_the_object(self):
        feats, first, truth = self._moving_square()
        soft = propagate_video(feats, first, n_context=3, size=4, topk=3, temperature=0.2)
        self.assertEqual(tuple(soft.shape), (8, 2, 12, 16))
        self.assertTrue(torch.equal(soft[0], first))
        self.assertTrue(torch.equal(soft.argmax(1), truth))

    def test_uninformative_features_lose_the_object(self):
        """Negative control: same video, features carry no object signal."""
        feats, first, truth = self._moving_square()
        noise = torch.randn(feats.shape, generator=torch.Generator().manual_seed(1))
        soft = propagate_video(noise, first, n_context=3, size=4, topk=3, temperature=0.2)
        pred = soft[-1].argmax(0).numpy() == 1
        self.assertLess(region_similarity(pred, truth[-1].numpy() == 1), 0.5)

    def test_neighbourhood_keeps_labels_off_a_far_lookalike(self):
        feats, first, truth = self._moving_square(w=30)
        feats[:, 4:7, 24:27] = feats[:, 4:7, 2:5][:1]  # same features, never labelled
        soft = propagate_video(feats, first, n_context=3, size=4, topk=3, temperature=0.2)
        self.assertEqual(int(soft.argmax(1)[:, 4:7, 24:27].sum()), 0)

    def test_labels_to_mask_normalisation(self):
        soft = torch.zeros(2, 4, 4)
        soft[0] = 0.7
        soft[1, 1:3, 1:3] = 0.3  # a faint object the background would out-vote
        self.assertEqual(int(labels_to_mask(soft, (4, 4), normalize=False).sum()), 0)
        self.assertGreater(int(labels_to_mask(soft, (4, 4), normalize=True).sum()), 0)
        self.assertEqual(tuple(labels_to_mask(soft, (40, 56)).shape), (40, 56))


class _PoolingEncoder(torch.nn.Module):
    """Stand-in encoder: a token is the mean pixel of its tubelet-patch, in
    the same token order as PatchEmbed3D (time, then rows, then columns)."""

    def forward(self, clip):
        return F.avg_pool3d(clip, (2, 16, 16)).flatten(2).transpose(1, 2)


class TestFrameFeatures(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.frames = torch.randn(7, 3, 32, 48)
        self.pooled = F.avg_pool2d(self.frames, 16).permute(0, 2, 3, 1)  # [N, h, w, 3]

    def test_frame_mode_gives_each_frame_its_own_map(self):
        feats = tracking.frame_features(_PoolingEncoder(), self.frames, "frame", batch_size=3)
        self.assertEqual(tuple(feats.shape), (7, 2, 3, 3))
        self.assertTrue(torch.allclose(feats, self.pooled, atol=1e-6))

    def test_clip_mode_shares_a_map_within_a_tubelet(self):
        feats = tracking.frame_features(_PoolingEncoder(), self.frames, "clip", frames_per_clip=4)
        self.assertEqual(tuple(feats.shape), (7, 2, 3, 3))
        for a in (0, 2, 4):
            pair = (self.pooled[a] + self.pooled[a + 1]) / 2
            self.assertTrue(torch.allclose(feats[a], pair, atol=1e-6))
            self.assertTrue(torch.equal(feats[a], feats[a + 1]))
        # last window is padded by holding the final frame
        self.assertTrue(torch.allclose(feats[6], self.pooled[6], atol=1e-6))

    def test_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            tracking.frame_features(_PoolingEncoder(), self.frames, "video")
        with self.assertRaises(ValueError):
            tracking.frame_features(_PoolingEncoder(), self.frames, "clip", frames_per_clip=3)

    def test_real_encoder_accepts_non_square_frames(self):
        cfg = yaml.safe_load(PROXY_BASE.read_text())
        cfg["model"]["model_name"] = "vit_tiny"
        encoder = build_encoder(cfg, 32, 2).eval()
        for mode in ("frame", "clip"):
            feats = tracking.frame_features(encoder, self.frames, mode, frames_per_clip=4)
            self.assertEqual(tuple(feats.shape[:3]), (7, 2, 3))
            self.assertTrue(bool(feats.isfinite().all()))


def _write_fake_davis(root, sequences=("alpha", "beta"), n_frames=6, h=64, w=96):
    """A DAVIS-layout folder: a bright square moving over a dark background."""
    (root / "ImageSets" / "2017").mkdir(parents=True)
    (root / "ImageSets" / "2017" / "val.txt").write_text("\n".join(sequences) + "\n")
    for s, name in enumerate(sequences):
        frame_dir = root / "JPEGImages" / "480p" / name
        mask_dir = root / "Annotations" / "480p" / name
        frame_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for t in range(n_frames):
            ids = np.zeros((h, w), dtype=np.uint8)
            ids[16:40, 8 + 6 * t : 32 + 6 * t] = 1
            if s == 1:  # second sequence has two objects
                ids[44:60, 60:80] = 2
            image = np.full((h, w, 3), 30, dtype=np.uint8)
            image[ids == 1] = (230, 60, 60)
            image[ids == 2] = (60, 60, 230)
            Image.fromarray(image).save(frame_dir / f"{t:05d}.jpg", quality=95)
            mask = Image.fromarray(ids, mode="P")
            mask.putpalette([0, 0, 0, 128, 0, 0, 0, 128, 0] + [0] * (253 * 3))
            mask.save(mask_dir / f"{t:05d}.png")


class TestDavisReader(unittest.TestCase):
    def test_reads_the_standard_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fake_davis(root)
            self.assertEqual(read_sequences(str(root)), ["alpha", "beta"])
            frames, masks = sequence_files(str(root), "beta")
            self.assertEqual((len(frames), len(masks)), (6, 6))
            ids = read_annotation(masks[0])
            self.assertEqual(sorted(np.unique(ids).tolist()), [0, 1, 2])
            tensor, original = load_frames(frames, short_side=64, patch_size=16)
            self.assertEqual(original, (64, 96))
            self.assertEqual(tuple(tensor.shape), (6, 3, 64, 96))

    def test_input_size_is_a_multiple_of_the_patch(self):
        self.assertEqual(model_input_size(480, 854, 480, 16), (480, 848))
        self.assertEqual(model_input_size(1080, 1920, 480, 16), (480, 848))
        self.assertEqual(model_input_size(854, 480, 480, 16), (848, 480))

    def test_rgb_annotation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.png"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
            with self.assertRaises(ValueError):
                read_annotation(str(path))


class TestEndToEnd(unittest.TestCase):
    """main() on a synthetic DAVIS folder and a saved (untrained) checkpoint."""

    def _run_folder(self, root):
        cfg = yaml.safe_load(PROXY_BASE.read_text())
        cfg["model"]["model_name"] = "vit_tiny"
        run = root / "run"
        run.mkdir()
        (run / "params-pretrain.yaml").write_text(yaml.safe_dump(cfg))
        torch.manual_seed(0)
        encoder = build_encoder(cfg, 64, 2)
        # train.py saves DistributedDataParallel(MultiSeqWrapper(vit))
        state = {f"module.backbone.{k}": v for k, v in encoder.state_dict().items()}
        torch.save({"target_encoder": state, "epoch": 3}, run / "latest.pth.tar")
        return run

    def _args(self, run, davis, **extra):
        return {
            "folder": str(run),
            "experiment": {
                "data": {"davis_root": str(davis), "short_side": 64, "frames_per_clip": 4},
                "propagation": {"n_context": 3, "neighborhood_size": 3},
                **extra,
            },
        }

    def test_report_for_both_modes_and_random_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            davis = root / "DAVIS"
            _write_fake_davis(davis)
            run = self._run_folder(root)
            for mode in ("frame", "clip"):
                report = tracking.main(self._args(run, davis, feature_mode=mode))
                self.assertEqual(report["epoch"], 3)
                self.assertEqual(len(report["per_object"]), 3)  # 1 + 2 objects
                overall = report["summary"]["all"]
                self.assertEqual(overall["objects"], 3)
                for key in ("J&F", "J", "F"):
                    self.assertGreaterEqual(overall[key], 0.0)
                    self.assertLessEqual(overall[key], 1.0)
                self.assertAlmostEqual(overall["J&F"], (overall["J"] + overall["F"]) / 2)
                out = run / "video_tracking_frozen" / "davis-tracking" / f"davis_val_{mode}_latest.json"
                self.assertEqual(json.loads(out.read_text())["settings"]["feature_mode"], mode)
            # the same weights give the same score: the checkpoint really is loaded
            again = tracking.main(self._args(run, davis, feature_mode="clip"))
            self.assertEqual(again["summary"], report["summary"])

            floor = tracking.main({**self._args(run, davis), "random_init": True})
            self.assertIsNone(floor["epoch"])
            self.assertEqual(floor["checkpoint"], "random-init")

    def test_mismatched_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            davis = root / "DAVIS"
            _write_fake_davis(davis)
            run = self._run_folder(root)
            checkpoint = torch.load(run / "latest.pth.tar")
            checkpoint["target_encoder"].pop(next(iter(checkpoint["target_encoder"])))
            torch.save(checkpoint, run / "latest.pth.tar")
            with self.assertRaises(RuntimeError):
                tracking.main(self._args(run, davis))


class TestScoring(unittest.TestCase):
    def test_summary_and_size_buckets(self):
        records = [
            {"sequence": "a", "object": 1, "area_fraction": 0.002, "J": 0.2, "F": 0.4},
            {"sequence": "a", "object": 2, "area_fraction": 0.03, "J": 0.6, "F": 0.8},
            {"sequence": "b", "object": 1, "area_fraction": 0.30, "J": 1.0, "F": 1.0},
        ]
        summary = tracking.summarize(records)
        self.assertAlmostEqual(summary["all"]["J"], 0.6)
        self.assertAlmostEqual(summary["all"]["F"], (0.4 + 0.8 + 1.0) / 3)
        self.assertAlmostEqual(summary["by_size"]["small"]["J&F"], 0.3)
        self.assertAlmostEqual(summary["by_size"]["medium"]["J&F"], 0.7)
        self.assertAlmostEqual(summary["by_size"]["large"]["J&F"], 1.0)
        self.assertIsNone(tracking.summarize([])["all"])

    def test_first_and_last_frames_are_not_scored(self):
        self.assertEqual(list(tracking.scored_frames(6)), [1, 2, 3, 4])
        self.assertEqual(list(tracking.scored_frames(2)), [])

    def test_perfect_features_score_perfectly_end_to_end(self):
        """evaluate_sequence with an encoder that can tell the object apart:
        the pooled-colour stand-in on the synthetic video must track it."""
        with tempfile.TemporaryDirectory() as tmp:
            davis = Path(tmp)
            _write_fake_davis(davis, sequences=("alpha",), n_frames=6)
            cfg = tracking.resolve_config(
                {"experiment": {"data": {"short_side": 64}, "propagation": {"n_context": 3, "neighborhood_size": 3}}}
            )
            cfg.update(patch_size=16, tubelet_size=2)
            rows = tracking.evaluate_sequence(
                _PoolingEncoder(), str(davis), "alpha", torch.device("cpu"), cfg
            )
            self.assertEqual(len(rows), 1)
            self.assertGreater(rows[0]["J"], 0.6)


class TestConfig(unittest.TestCase):
    def test_defaults_are_the_paper_protocol(self):
        cfg = tracking.resolve_config({})
        self.assertEqual(
            (cfg["short_side"], cfg["n_context"], cfg["neighborhood_size"],
             cfg["neighborhood_shape"], cfg["topk"], cfg["temperature"]),
            (480, 15, 12, "circle", 5, 0.2),
        )
        with self.assertRaises(ValueError):
            tracking.resolve_config({"experiment": {"dtype": "float64"}})

    def test_shipped_config_matches_the_defaults(self):
        args = yaml.safe_load(EVAL_CONFIG.read_text())
        self.assertEqual(args["eval_name"], "video_tracking_frozen")
        self.assertEqual(args["checkpoint_key"], "target_encoder")
        shipped = tracking.resolve_config(args)
        default = tracking.resolve_config({})
        shipped.pop("davis_root"), default.pop("davis_root")
        self.assertEqual(shipped, default)


if __name__ == "__main__":
    unittest.main()
