"""
Local-density estimation used by LD-AMFCM.

Two estimators appear in the original notebook:

``mean_gaussian_global_median``  (headline configuration)
    sigma is the median of *all* kNN distances of the matrix, i.e. a single
    global scalar, and rho_i is the mean of the Gaussian kernels of the k
    nearest neighbours of i.

``sum_gaussian_perpoint_mean``   (variant, m_min / m_max sweeps only)
    sigma_i is the mean distance to the k nearest neighbours of i, i.e. a
    per-point scalar, and rho_i is the *sum* of the Gaussian kernels of those
    neighbours.

The two are not interchangeable: they give different density values, and
therefore different weights, different sample-wise fuzzifiers and different
clusterings.  The variant is reproduced because those sweeps were produced
with it, and every output produced through it is flagged.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

MAIN_ESTIMATOR = "mean_gaussian_global_median"
VARIANT_ESTIMATOR = "sum_gaussian_perpoint_mean"

ESTIMATORS = (MAIN_ESTIMATOR, VARIANT_ESTIMATOR)


def local_density(X, k=12, estimator=MAIN_ESTIMATOR, algorithm="auto", n_jobs=1):
    """Raw local density rho for every sample.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Standardised data, *including* the injected contamination.
    k : int
        Neighbourhood size.  The k nearest neighbours excluding the point
        itself are used.
    estimator : str
        One of :data:`ESTIMATORS`.
    algorithm, n_jobs :
        Passed through to :class:`sklearn.neighbors.NearestNeighbors`.  The
        scalability experiment uses ``algorithm="brute"`` with
        ``n_jobs=-1``; every other experiment leaves the defaults.

    Returns
    -------
    rho : ndarray, shape (n_samples,)
        Unnormalised local density.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(
            "Unknown density estimator {!r}; expected one of {}".format(
                estimator, ESTIMATORS
            )
        )

    n_samples = X.shape[0]
    n_neighbors = min(k + 1, n_samples)

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric="euclidean",
        n_jobs=n_jobs,
    )
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Remove the self-distance.
    distances = distances[:, 1:]

    if distances.shape[1] == 0:
        raise ValueError("No nearest-neighbour distances available.")

    if estimator == MAIN_ESTIMATOR:
        sigma = np.median(distances)
        if sigma <= 0:
            sigma = 1.0
        rho = np.exp(-(distances ** 2) / (sigma ** 2)).mean(axis=1)
    else:  # VARIANT_ESTIMATOR
        sigma = np.mean(distances, axis=1) + 1e-12
        rho = np.sum(
            np.exp(-(distances ** 2) / (2.0 * sigma[:, None] ** 2)),
            axis=1,
        )

    return rho


def normalize_density(rho):
    """Min-max normalisation of the density to [0, 1].

    The additive 1e-12 guard makes a degenerate density collapse to zero
    rather than raising.  The original scalability cell used a separate guard
    that returns a constant 0.5 in that case; the difference is documented in
    docs/KNOWN_VARIATIONS.md and never occurs for the reported benchmarks.
    """
    rho_min = np.min(rho)
    rho_max = np.max(rho)
    return (rho - rho_min) / (rho_max - rho_min + 1e-12)


def density_weights(rho):
    """Normalised sample weights w_i = rho_i / sum(rho)."""
    rho_sum = np.sum(rho)
    if rho_sum <= 1e-12:
        return np.ones_like(rho) / len(rho)
    return rho / rho_sum


def measure_density_time(X, k=12, estimator=MAIN_ESTIMATOR, algorithm="auto", n_jobs=1):
    """Return ``(rho, elapsed_seconds)`` for the scalability study."""
    import time

    t_start = time.perf_counter()
    rho = local_density(X, k=k, estimator=estimator, algorithm=algorithm, n_jobs=n_jobs)
    return rho, time.perf_counter() - t_start
