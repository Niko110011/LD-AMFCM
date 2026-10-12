"""
Evaluation metrics.

All clustering metrics are computed on the uncontaminated samples only, using
the same partition that the method produced over the full augmented matrix.
The two noise-detection scores use all samples, because their whole purpose is
to test whether the injected points can be told apart.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from .config import NOISE_LABEL


# ----------------------------------------------------------------------
# Clustering agreement
# ----------------------------------------------------------------------

def clean_ari(y_true, labels):
    """Adjusted Rand index on the uncontaminated samples only."""
    mask = y_true >= 0
    return adjusted_rand_score(y_true[mask], labels[mask])


def davies_bouldin_index(X, labels):
    """Davies-Bouldin index of the given partition.

    Returns 0 when fewer than two clusters are present, matching the original
    implementation.  Callers that must distinguish a degenerate partition use
    :func:`safe_davies_bouldin_index` instead.
    """
    clusters = np.unique(labels)
    clusters = clusters[clusters >= 0]
    k = len(clusters)
    if k < 2:
        return 0

    centroids = np.array([X[labels == c].mean(axis=0) for c in clusters])
    scatter = np.array([
        np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1))
        for i, c in enumerate(clusters)
    ])

    ratio = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                dist = np.linalg.norm(centroids[i] - centroids[j])
                ratio[i, j] = (scatter[i] + scatter[j]) / (dist + 1e-12)
    return np.mean(np.max(ratio, axis=1))


def safe_davies_bouldin_index(X, labels):
    """DBI as reported in the comparison table: NaN for degenerate partitions."""
    if len(np.unique(labels)) >= 2:
        return davies_bouldin_index(X, labels)
    return np.nan


def clean_dbi(X, y_true, labels):
    """DBI restricted to the uncontaminated samples."""
    mask = y_true >= 0
    return safe_davies_bouldin_index(X[mask], labels[mask])


# ----------------------------------------------------------------------
# Noise diagnostics
# ----------------------------------------------------------------------

def noise_membership(U, y_true):
    """Mean of the largest membership over the injected points.

    Lower is better: the closer to zero, the more confidently the method has
    placed the contamination on the periphery of some cluster.
    """
    noise_mask = y_true < 0
    if not np.any(noise_mask):
        return 0.0
    return float(np.mean(U[noise_mask].max(axis=1)))


def noise_auc(U, y_true):
    """ROC-AUC of the membership uncertainty against the injected-point flag.

    The score is ``1 - max_k u_ik``, so 0.5 is chance level and 1.0 means the
    contamination is ranked as more uncertain than every clean sample.
    """
    is_noise = (y_true < 0).astype(int)
    if not np.any(is_noise):
        return 0.5
    return float(roc_auc_score(is_noise, 1.0 - U.max(axis=1)))


def density_auc(rho, y_true):
    """ROC-AUC of the local density against the injected-point flag.

    This is a property of the density estimator alone and is independent of
    any clustering algorithm.
    """
    is_noise = (y_true < 0).astype(int)
    if not np.any(is_noise):
        return 0.5
    return float(roc_auc_score(is_noise, -rho))


def density_ratio(rho, y_true):
    """Median density of the injected points divided by that of the clean ones."""
    clean = y_true >= 0
    noise = y_true < 0
    if not np.any(noise):
        return 0.0
    return float(np.median(rho[noise]) / (np.median(rho[clean]) + 1e-12))


# ----------------------------------------------------------------------
# Cluster-level recall after optimal label matching
# ----------------------------------------------------------------------

def best_cluster_mapping(y_true, y_pred, n_clusters):
    """Match predicted cluster ids to true ids by the Hungarian algorithm."""
    contingency = np.zeros((n_clusters, n_clusters), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_clusters and 0 <= p < n_clusters:
            contingency[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    return {pred: true for true, pred in zip(row_ind, col_ind)}


def cluster_recalls(y_true, y_pred, n_clusters):
    """Per-cluster recall after optimal matching, plus min and macro average."""
    mask = y_true >= 0
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)

    mapping = best_cluster_mapping(yt, yp, n_clusters)
    yp_mapped = np.array([mapping.get(x, -1) for x in yp])

    recalls = {}
    for c in range(n_clusters):
        idx = yt == c
        recalls[c] = np.nan if idx.sum() == 0 else float(np.mean(yp_mapped[idx] == c))

    valid = [v for v in recalls.values() if not np.isnan(v)]
    return recalls, float(np.min(valid)), float(np.mean(valid))


# ----------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------

def mean_std(values, ddof=1):
    """Mean and sample standard deviation, ignoring NaNs where all are NaN."""
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values, ddof=ddof))


def nanmean_std(values, ddof=1):
    """Mean and standard deviation ignoring NaN entries."""
    values = np.asarray(values, dtype=float)
    if np.all(np.isnan(values)):
        return np.nan, np.nan
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=ddof))
