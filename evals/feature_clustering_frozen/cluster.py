# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Clusterers for the frozen-feature clustering probe.

Two interchangeable functions, both `(x, k, seed) -> (centroids, labels)`:

  lloyd_kmeans   plain k-means (k-means++ init, Lloyd iterations), in torch.
                 The reference, and the path that needs no extra dependency.

  emeans         e-means (Meta FAIR, Nov 2026), through Faiss >= 1.16
                 `faiss.contrib.e_means`.  It replaces Lloyd's full passes
                 with mini-batch exponential-moving-average updates and is
                 reported by its authors to reach Lloyd's 25-epoch loss in
                 1.2-3 epochs.

WHAT TO EXPECT FROM SWAPPING ONE FOR THE OTHER
----------------------------------------------
Both minimise the same objective (mean squared distance to the assigned
centroid), so on a probe-sized feature set they should land at about the same
`quantization_error`, purity and NMI.  The claimed advantage of e-means is
TIME.  It matters when clustering millions of patch tokens, not a few
thousand pooled clip vectors.  The probe reports `quantization_error` and
wall time for both so that claim can be checked on your own features rather
than taken from the paper.

STATUS OF THE e-means ADAPTER
-----------------------------
Written against the README of faiss/contrib/e_means (`NumpyEmeans(d, k)` /
`Emeans(d, k)`, `.train(x) -> (centroids, labels)`).  Faiss was not installed
where this was written, so the adapter has NOT been executed; its test is
skipped until `faiss.contrib.e_means` imports.
"""

import torch


def _sq_dists(x, centroids, chunk=65536):
    """Squared Euclidean distances [N, k], computed in row chunks."""
    c_sq = (centroids**2).sum(1)
    out = []
    for start in range(0, x.size(0), chunk):
        xb = x[start : start + chunk]
        d = (xb**2).sum(1, keepdim=True) - 2.0 * xb @ centroids.T + c_sq
        out.append(d.clamp_min_(0.0))
    return torch.cat(out)


def _kmeans_pp_init(x, k, gen):
    """k-means++ seeding: each new centroid is drawn with probability
    proportional to its squared distance from the centroids chosen so far."""
    n = x.size(0)
    first = torch.randint(n, (1,), generator=gen, device=x.device)
    centroids = [x[first.item()]]
    closest = _sq_dists(x, centroids[0][None]).squeeze(1)
    for _ in range(1, k):
        total = closest.sum()
        if total <= 0:  # every point already coincides with a centroid
            idx = torch.randint(n, (1,), generator=gen, device=x.device)
        else:
            idx = torch.multinomial(closest / total, 1, generator=gen)
        centroids.append(x[idx.item()])
        closest = torch.minimum(closest, _sq_dists(x, centroids[-1][None]).squeeze(1))
    return torch.stack(centroids)


@torch.no_grad()
def lloyd_kmeans(x, k, seed=0, iters=25):
    """Plain k-means.  x: [N, D] tensor.  Returns (centroids [k, D], labels [N])."""
    x = x.detach().float()
    if x.ndim != 2:
        raise ValueError(f"expected [N, D], got shape {tuple(x.shape)}")
    if not 0 < k <= x.size(0):
        raise ValueError(f"k={k} must be in [1, N={x.size(0)}]")
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    centroids = _kmeans_pp_init(x, k, gen)

    labels = None
    for _ in range(iters):
        dists = _sq_dists(x, centroids)
        new_labels = dists.argmin(1)
        if labels is not None and torch.equal(new_labels, labels):
            break  # converged: assignments stopped changing
        labels = new_labels

        sums = torch.zeros_like(centroids).index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k)
        occupied = counts > 0
        centroids[occupied] = sums[occupied] / counts[occupied, None].float()
        # An empty cluster would otherwise stay empty forever.  Re-seed it at
        # the point currently worst served by its own centroid.
        if not occupied.all():
            worst = dists.gather(1, labels[:, None]).squeeze(1)
            order = worst.argsort(descending=True)
            for j, slot in enumerate((~occupied).nonzero().flatten().tolist()):
                centroids[slot] = x[order[j]]
    labels = _sq_dists(x, centroids).argmin(1)
    return centroids, labels


@torch.no_grad()
def emeans(x, k, seed=0):
    """e-means through Faiss.  Same contract as `lloyd_kmeans`.

    Uses the PyTorch implementation when `x` is on CUDA and the NumPy one
    otherwise.  `seed` seeds NumPy / torch globally just before training,
    because the README shows no seed argument on the constructors.
    """
    try:
        from faiss.contrib import e_means as faiss_emeans
    except ImportError as exc:
        raise ImportError(
            "The e-means clusterer needs Faiss >= 1.16 (faiss.contrib.e_means). "
            "Install it, or drop 'emeans' from experiment.clustering.clusterers."
        ) from exc

    x = x.detach().float()
    dim = x.size(1)
    if x.is_cuda:
        torch.manual_seed(seed)
        centroids, labels = faiss_emeans.Emeans(dim, k).train(x)
    else:
        import numpy as np

        np.random.seed(seed)
        centroids, labels = faiss_emeans.NumpyEmeans(dim, k).train(
            np.ascontiguousarray(x.numpy())
        )
    centroids = torch.as_tensor(centroids, dtype=torch.float32, device=x.device)
    labels = torch.as_tensor(labels, device=x.device).long().flatten()
    return centroids, labels


CLUSTERERS = {
    "kmeans": lloyd_kmeans,
    "emeans": emeans,
}


@torch.no_grad()
def quantization_error(x, centroids, labels):
    """Mean squared distance to the assigned centroid: the objective both
    clusterers minimise, so the fair like-for-like comparison between them."""
    x = x.detach().float()
    return float(((x - centroids[labels]) ** 2).sum(1).mean())
