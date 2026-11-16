# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Label-agreement metrics for a clustering.  NumPy only (no scikit-learn).

Both take integer arrays of equal length: the cluster id each sample was
assigned, and its ground-truth class.  Labels are used ONLY here, to score the
clustering after the fact -- never to produce it.

  purity   For each cluster take its most common class; purity is the
           fraction of samples that belong to their cluster's majority class.
           Easy to read, but it rises with the number of clusters (k = N
           gives 1.0), so compare it only at equal k.

  nmi      Normalised mutual information, I(C; Y) / mean(H(C), H(Y)), in
           [0, 1].  Corrects for the number of clusters, so it is the one to
           quote across settings.  Same normalisation as scikit-learn's
           default ("arithmetic").
"""

import numpy as np


def _contingency(clusters, classes):
    clusters = np.asarray(clusters).ravel()
    classes = np.asarray(classes).ravel()
    if clusters.shape != classes.shape:
        raise ValueError(f"length mismatch: {clusters.shape} vs {classes.shape}")
    if clusters.size == 0:
        raise ValueError("empty input")
    _, ci = np.unique(clusters, return_inverse=True)
    _, yi = np.unique(classes, return_inverse=True)
    table = np.zeros((ci.max() + 1, yi.max() + 1), dtype=np.int64)
    np.add.at(table, (ci, yi), 1)
    return table


def purity(clusters, classes):
    table = _contingency(clusters, classes)
    return float(table.max(axis=1).sum() / table.sum())


def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def nmi(clusters, classes):
    table = _contingency(clusters, classes)
    n = table.sum()
    row, col = table.sum(axis=1), table.sum(axis=0)
    h_c, h_y = _entropy(row), _entropy(col)
    if h_c == 0.0 or h_y == 0.0:
        # one side is a single group: no information can be shared
        return 1.0 if (h_c == 0.0 and h_y == 0.0) else 0.0
    nz = table > 0
    joint = table[nz] / n
    expected = (row[:, None] * col[None, :])[nz] / (n * n)
    mutual_info = float((joint * np.log(joint / expected)).sum())
    return max(0.0, mutual_info / (0.5 * (h_c + h_y)))
