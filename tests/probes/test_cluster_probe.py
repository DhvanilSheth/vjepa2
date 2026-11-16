# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Stage-8 gates: the frozen-feature clustering probe.

Covers the pure pieces.  `eval.main()` (dataloader + real checkpoint) is not
exercised here -- it needs decord, a dataset and a trained checkpoint.
"""

import ast
import unittest
from pathlib import Path

import numpy as np
import torch

from evals.feature_clustering_frozen.cluster import (
    CLUSTERERS,
    emeans,
    lloyd_kmeans,
    quantization_error,
)
from evals.feature_clustering_frozen.metrics import nmi, purity

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    from faiss.contrib import e_means as _faiss_emeans  # noqa: F401

    HAS_EMEANS = True
except ImportError:
    HAS_EMEANS = False


def _blobs(n_per=200, k=5, d=16, spread=0.05, seed=0):
    """k well-separated Gaussian blobs; returns (x, true_labels)."""
    gen = torch.Generator().manual_seed(seed)
    centers = torch.randn(k, d, generator=gen) * 5.0
    labels = torch.arange(k).repeat_interleave(n_per)
    x = centers[labels] + spread * torch.randn(k * n_per, d, generator=gen)
    perm = torch.randperm(x.size(0), generator=gen)
    return x[perm], labels[perm]


class TestMetrics(unittest.TestCase):
    def test_perfect_clustering_up_to_relabelling(self):
        classes = np.array([0, 0, 1, 1, 2, 2])
        clusters = np.array([7, 7, 3, 3, 9, 9])  # same partition, other names
        self.assertEqual(purity(clusters, classes), 1.0)
        self.assertAlmostEqual(nmi(clusters, classes), 1.0, places=9)

    def test_hand_computed_values(self):
        classes = np.array([0, 0, 0, 1, 1, 1])
        clusters = np.array([0, 0, 1, 1, 1, 1])
        # cluster 0 -> {0,0}; cluster 1 -> {0,1,1,1}: (2 + 3) / 6
        self.assertAlmostEqual(purity(clusters, classes), 5 / 6)
        h_y = np.log(2)
        h_c = -(2 / 6 * np.log(2 / 6) + 4 / 6 * np.log(4 / 6))
        mi = (
            2 / 6 * np.log((2 / 6) / (2 / 6 * 3 / 6))
            + 1 / 6 * np.log((1 / 6) / (4 / 6 * 3 / 6))
            + 3 / 6 * np.log((3 / 6) / (4 / 6 * 3 / 6))
        )
        self.assertAlmostEqual(nmi(clusters, classes), mi / (0.5 * (h_c + h_y)), places=9)

    def test_unrelated_labels_score_near_zero_nmi(self):
        rng = np.random.default_rng(0)
        classes = rng.integers(0, 10, size=20000)
        clusters = rng.integers(0, 10, size=20000)
        self.assertLess(nmi(clusters, classes), 0.01)

    def test_purity_is_inflated_by_overclustering_but_nmi_is_not(self):
        """Why NMI is the number to quote across different k."""
        rng = np.random.default_rng(0)
        classes = rng.integers(0, 4, size=2000)
        singletons = np.arange(2000)
        self.assertEqual(purity(singletons, classes), 1.0)
        self.assertLess(nmi(singletons, classes), 0.35)

    def test_degenerate_and_bad_inputs(self):
        self.assertEqual(nmi(np.zeros(10), np.arange(10) % 2), 0.0)
        self.assertEqual(nmi(np.zeros(10), np.zeros(10)), 1.0)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            purity(np.zeros(3), np.zeros(4))
        with self.assertRaisesRegex(ValueError, "empty"):
            nmi(np.array([]), np.array([]))


class TestLloydKMeans(unittest.TestCase):
    def test_recovers_separated_blobs(self):
        x, y = _blobs()
        for seed in range(5):
            with self.subTest(seed=seed):
                centroids, labels = lloyd_kmeans(x, 5, seed=seed)
                self.assertEqual(centroids.shape, (5, 16))
                self.assertEqual(purity(labels.numpy(), y.numpy()), 1.0)
                self.assertAlmostEqual(nmi(labels.numpy(), y.numpy()), 1.0, places=6)

    def test_objective_matches_a_hand_computation_and_beats_random(self):
        x, _ = _blobs()
        centroids, labels = lloyd_kmeans(x, 5, seed=0)
        by_hand = sum(
            ((x[labels == j] - centroids[j]) ** 2).sum() for j in range(5)
        ) / x.size(0)
        self.assertAlmostEqual(quantization_error(x, centroids, labels), float(by_hand), places=4)
        # d * spread^2 = 16 * 0.0025 is the within-blob variance
        self.assertAlmostEqual(quantization_error(x, centroids, labels), 0.04, delta=0.01)
        # seed != _blobs' seed: the same seed would reproduce the true centres
        gen = torch.Generator().manual_seed(99)
        random_centroids = torch.randn(5, 16, generator=gen) * 5.0
        random_labels = torch.cdist(x, random_centroids).argmin(1)
        self.assertGreater(
            quantization_error(x, random_centroids, random_labels),
            10 * quantization_error(x, centroids, labels),
        )

    def test_deterministic_per_seed(self):
        x, _ = _blobs(spread=2.0)  # overlapping, so the seed matters
        a = lloyd_kmeans(x, 8, seed=3)
        b = lloyd_kmeans(x, 8, seed=3)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        self.assertTrue(torch.equal(a[1], b[1]))

    def test_no_empty_clusters_even_with_duplicates(self):
        x = torch.cat([torch.zeros(50, 4), torch.ones(50, 4), torch.randn(20, 4)])
        _, labels = lloyd_kmeans(x, 6, seed=0)
        self.assertEqual(labels.unique().numel(), 6)

    def test_rejects_bad_arguments(self):
        with self.assertRaisesRegex(ValueError, "must be in"):
            lloyd_kmeans(torch.randn(4, 2), 5)
        with self.assertRaisesRegex(ValueError, r"expected \[N, D\]"):
            lloyd_kmeans(torch.randn(4, 2, 2), 2)


class TestEmeansAdapter(unittest.TestCase):
    def test_registered(self):
        self.assertEqual(set(CLUSTERERS), {"kmeans", "emeans"})

    @unittest.skipIf(HAS_EMEANS, "Faiss e-means is installed")
    def test_missing_faiss_gives_an_actionable_error(self):
        with self.assertRaisesRegex(ImportError, r"Faiss >= 1\.16"):
            emeans(torch.randn(10, 4), 2)

    @unittest.skipUnless(HAS_EMEANS, "needs Faiss >= 1.16 (faiss.contrib.e_means)")
    def test_emeans_matches_kmeans_on_separated_blobs(self):
        x, y = _blobs()
        centroids, labels = emeans(x, 5, seed=0)
        self.assertEqual(centroids.shape, (5, 16))
        self.assertEqual(labels.shape, (x.size(0),))
        self.assertEqual(purity(labels.numpy(), y.numpy()), 1.0)
        ref = quantization_error(x, *lloyd_kmeans(x, 5, seed=0))
        self.assertLess(quantization_error(x, centroids, labels), 1.5 * ref)


class TestProbeModelPlumbing(unittest.TestCase):
    CFG = {
        "data": {"patch_size": 16, "tubelet_size": 2},
        "meta": {"use_sdpa": True},
        "model": {
            "model_name": "vit_tiny",
            "use_rope": True,
            "img_temporal_dim_size": 1,
            "modality_embedding": True,
            "uniform_power": True,
            "interpolate_rope": True,
        },
    }

    def _cfg(self, **model_overrides):
        cfg = {k: dict(v) for k, v in self.CFG.items()}
        cfg["model"].update(model_overrides)
        return cfg

    def test_probe_covers_every_encoder_kwarg_of_init_video_model(self):
        """A new architecture kwarg added to the training path must be added
        to the probe too, or the probe silently builds a different model."""
        from evals.feature_clustering_frozen.eval import encoder_kwargs_from_pretrain

        tree = ast.parse((REPO_ROOT / "app" / "vjepa_3" / "utils.py").read_text())
        train_kwargs = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Subscript)
                and ast.unparse(node.func) == "video_vit.__dict__[model_name]"
            ):
                train_kwargs = {kw.arg for kw in node.keywords}
        self.assertIsNotNone(train_kwargs, "encoder factory call not found in utils.py")
        probe_kwargs = set(encoder_kwargs_from_pretrain(self.CFG, 64, 8))
        self.assertEqual(probe_kwargs, train_kwargs)

    def test_architecture_flags_are_honoured(self):
        from app.vjepa_3.models.utils.ffn import SquaredReLU
        from app.vjepa_3.models.utils.norms import DynamicErf
        from evals.feature_clustering_frozen.eval import build_encoder

        enc = build_encoder(
            self._cfg(ffn_type="relu2", norm_type="derf", use_string=True, rope_theta=100.0),
            resolution=64,
            frames_per_clip=8,
        )
        self.assertIsInstance(enc.blocks[0].mlp.act, SquaredReLU)
        self.assertIsInstance(enc.blocks[0].norm1, DynamicErf)
        self.assertIsNotNone(enc.string)
        self.assertIsNotNone(enc.blocks[0].attn.omega_d)

    def test_checkpoint_round_trip_is_strict(self):
        from evals.feature_clustering_frozen.eval import (
            build_encoder,
            load_encoder_weights,
            pool_clip_features,
        )

        cfg = self._cfg(norm_type="dyt", use_string=True, string_init_std=0.1)
        torch.manual_seed(0)
        trained = build_encoder(cfg, 64, 8).eval()
        # the key layout train.py writes: DDP(MultiSeqWrapper(vit))
        saved = {f"module.backbone.{k}": v for k, v in trained.state_dict().items()}

        torch.manual_seed(1)
        fresh = build_encoder(cfg, 64, 8).eval()
        load_encoder_weights(fresh, saved)
        clips = [[torch.randn(2, 3, 8, 64, 64)]]
        torch.testing.assert_close(
            pool_clip_features(trained, clips), pool_clip_features(fresh, clips), rtol=0, atol=0
        )

        # a checkpoint from a DIFFERENT architecture must be refused, not
        # partially loaded
        other = build_encoder(self._cfg(), 64, 8)
        with self.assertRaises(RuntimeError):
            load_encoder_weights(other, saved)

    def test_pooling_averages_tokens_segments_and_views(self):
        from evals.feature_clustering_frozen.eval import pool_clip_features

        class Fake(torch.nn.Module):
            def forward(self, x):  # [B, N, D] tokens whose value is the input mean
                return x.mean(dim=(1, 2, 3, 4))[:, None, None].expand(-1, 5, 3)

        a, b, c = (torch.full((2, 3, 4, 8, 8), v) for v in (1.0, 2.0, 6.0))
        out = pool_clip_features(Fake(), [[a, b], [c]])
        self.assertEqual(out.shape, (2, 3))
        torch.testing.assert_close(out, torch.full((2, 3), 3.0))

    def test_score_clusterings_end_to_end(self):
        from evals.feature_clustering_frozen.eval import score_clusterings

        x, y = _blobs()
        report = score_clusterings(x, y, ["kmeans"], k=5)
        self.assertEqual(report["kmeans"]["purity"], 1.0)
        self.assertEqual(report["kmeans"]["k"], 5)
        self.assertEqual(
            set(report["kmeans"]), {"purity", "nmi", "quantization_error", "seconds", "k"}
        )
        with self.assertRaisesRegex(ValueError, "Unknown clusterer"):
            score_clusterings(x, y, ["dbscan"], k=5)


if __name__ == "__main__":
    unittest.main()
