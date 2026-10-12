"""
Benchmark construction.

Every benchmark is a standard, publicly available data set with a known
partition, augmented with artificially injected contamination whose points are
labelled ``-1`` and are excluded from every clustering metric.

Two seeding conventions appear in the original notebook and both are needed to
reproduce the published numbers:

``local_rng=True``
    Each benchmark draws its contamination from a freshly seeded generator
    ``RandomState(42)``.  This is what the single-benchmark sensitivity cells
    did, where one benchmark was generated per executed cell.

``local_rng=False``
    All benchmarks draw from one shared stream that is seeded once, in the
    declared order of :data:`ldamfcm.config.BENCHMARK_ORDER`.  This is what the
    comparison, ablation and regenerated-input studies did.

The same benchmark can therefore carry *different* contaminating points under
the two conventions.  They are separate protocols and are reported separately.

Standardisation is always a single :class:`sklearn.preprocessing.StandardScaler`
fitted on the augmented matrix (clean samples plus injected points) and applied
before any density estimation.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

from . import config

NOISE_LABEL = config.NOISE_LABEL


# ----------------------------------------------------------------------
# Individual benchmark generators
# ----------------------------------------------------------------------

def generate_blobs(rng):
    """Blobs + uniform noise: 400 clean samples, 3 clusters, 60 noise points."""
    X_clean, y_clean = make_blobs(
        n_samples=400,
        centers=3,
        cluster_std=[0.6, 1.0, 1.4],
        random_state=1,
    )
    noise = rng.uniform(-6, 6, (60, 2))
    X = np.vstack([X_clean, noise])
    y = np.hstack([y_clean, NOISE_LABEL * np.ones(60)])
    return "Blobs + Uniform Noise", X, y


def generate_iris(rng):
    """Iris + outliers: 150 clean samples, 3 clusters, 20 noise points."""
    iris = load_iris()
    X_clean, y_clean = iris.data, iris.target

    outliers = (
        rng.uniform(X_clean.min(axis=0), X_clean.max(axis=0), (20, 4)) * 1.5
    )
    X = np.vstack([X_clean, outliers])
    y = np.hstack([y_clean, NOISE_LABEL * np.ones(20)])
    return "Iris + Outliers", X, y


def generate_moons(rng):
    """Two-Moons + Gaussian noise: 300 clean samples, 2 clusters, 40 noise points."""
    X_clean, y_clean = make_moons(n_samples=300, noise=0.1, random_state=42)
    noise = rng.normal(0.0, 1.5, (40, 2))
    X = np.vstack([X_clean, noise])
    y = np.hstack([y_clean, NOISE_LABEL * np.ones(40)])
    return "Moons + Gaussian Noise", X, y


def generate_anisotropic(rng):
    """Anisotropic blobs + noise: 300 clean samples, 3 clusters, 50 noise points."""
    X_clean, y_clean = make_blobs(n_samples=300, centers=3, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_clean = np.dot(X_clean, transformation)

    noise = rng.uniform(X_clean.min(), X_clean.max(), (50, 2))
    X = np.vstack([X_clean, noise])
    y = np.hstack([y_clean, NOISE_LABEL * np.ones(50)])
    return "Anisotropic + Noise", X, y


def generate_breast(rng):
    """Breast Cancer WDBC + outliers: 569 clean samples, 2 clusters, 30 noise points."""
    breast = load_breast_cancer()
    X_clean, y_clean = breast.data, breast.target

    n_outliers = 30
    outliers = rng.uniform(
        X_clean.min(axis=0), X_clean.max(axis=0),
        (n_outliers, X_clean.shape[1]),
    )
    X = np.vstack([X_clean, outliers])
    y = np.hstack([y_clean, NOISE_LABEL * np.ones(n_outliers)])
    return "Breast Cancer + Outliers", X, y


_BENCHMARK_GENERATORS = {
    "Blobs + Uniform Noise": generate_blobs,
    "Iris + Outliers": generate_iris,
    "Moons + Gaussian Noise": generate_moons,
    "Anisotropic + Noise": generate_anisotropic,
    "Breast Cancer + Outliers": generate_breast,
}


# ----------------------------------------------------------------------
# Benchmark collection
# ----------------------------------------------------------------------

def get_benchmarks(data_seed=42, local_rng=False):
    """Build the five augmented benchmarks.

    Returns
    -------
    list of ``(name, X_raw, y_true)`` in :data:`config.BENCHMARK_ORDER`.
        ``X_raw`` is unstandardised; call :func:`standardize` before use.
    """
    if local_rng:
        benchmarks = []
        for name in config.BENCHMARK_ORDER:
            rng = np.random.RandomState(data_seed)
            benchmarks.append(_BENCHMARK_GENERATORS[name](rng))
        return benchmarks

    np.random.seed(data_seed)
    return [
        _BENCHMARK_GENERATORS[name](np.random)
        for name in config.BENCHMARK_ORDER
    ]


def get_benchmark(name, data_seed=42, local_rng=True):
    """Build one benchmark on its own.

    ``local_rng=True`` seeds a fresh generator for this benchmark, which is
    what the single-benchmark sensitivity cells did.

    ``local_rng=False`` seeds the shared stream and then generates only this
    benchmark, so its contamination corresponds to this benchmark being
    *first* in :data:`config.BENCHMARK_ORDER`.  That is **not** the same data
    as in the comparison study, where it is preceded by other benchmarks.  Use
    :func:`get_benchmarks` with ``local_rng=False`` whenever the published
    comparison numbers are the target.
    """
    if local_rng:
        rng = np.random.RandomState(data_seed)
    else:
        np.random.seed(data_seed)
        rng = np.random
    return _BENCHMARK_GENERATORS[name](rng)


def standardize(X_raw):
    """Z-score standardisation fitted on the augmented matrix."""
    return StandardScaler().fit_transform(X_raw)


def n_clusters_from_labels(y_true):
    """Number of legitimate clusters, i.e. distinct labels other than -1."""
    return len(np.unique(y_true[y_true >= 0]))


def clean_mask(y_true):
    """Boolean mask selecting the uncontaminated samples."""
    return y_true >= 0


def contamination_ratio(y_true, denominator="augmented"):
    """Contamination ratio together with its denominator convention.

    ``"augmented"`` : n_noise / (n_clean + n_noise)
    ``"clean"``     : n_noise / n_clean
    """
    n_noise = int(np.sum(y_true < 0))
    n_clean = int(np.sum(y_true >= 0))

    if denominator == "augmented":
        return n_noise / float(n_noise + n_clean)
    if denominator == "clean":
        return n_noise / float(n_clean)
    raise ValueError("Unknown denominator convention {!r}".format(denominator))


# ----------------------------------------------------------------------
# Scalability data
# ----------------------------------------------------------------------

def generate_scalability_data(n, p, n_clusters=config.N_CLUSTERS_SCALABILITY,
                              seed=config.SCALABILITY_GENERATOR_SEED):
    """Clean synthetic data for the scalability study.

    The data are held fixed across the repetitions of that study, so the data
    seed is constant.  Only the first two features carry cluster structure;
    the remaining ``p - 2`` features are pure noise dimensions.
    """
    rng = np.random.RandomState(seed)

    base_size = n // n_clusters
    remainder = n % n_clusters
    cluster_sizes = [
        base_size + (1 if i < remainder else 0) for i in range(n_clusters)
    ]

    if n_clusters == 3:
        centers_2d = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    else:
        angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
        centers_2d = np.column_stack(
            [5.0 * np.cos(angles), 5.0 * np.sin(angles)]
        )

    X_list, y_list = [], []
    for c in range(n_clusters):
        nc = cluster_sizes[c]
        X_signal = centers_2d[c] + rng.normal(0.0, 1.0, (nc, 2))

        if p > 2:
            X_noise = rng.normal(0.0, 1.0, (nc, p - 2))
            X_cluster = np.hstack([X_signal, X_noise])
        else:
            X_cluster = X_signal[:, :p]

        X_list.append(X_cluster)
        y_list.append(np.full(nc, c, dtype=int))

    return np.vstack(X_list), np.concatenate(y_list)


# ----------------------------------------------------------------------
# Density-reliability stress scenarios
# ----------------------------------------------------------------------

def generate_heterogeneous_density(seed=42):
    """Scenario A: equal-size legitimate clusters with very different spread.

    Cluster standard deviations 0.45 / 0.9 / 1.5 with 180 samples each, so
    local density differs strongly between legitimate clusters.
    """
    rng = np.random.RandomState(seed)

    centers = np.array([[-4.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
    stds = [0.45, 0.9, 1.5]
    n_per_cluster = 180

    X_clean, y_clean = [], []
    for c, (center, std) in enumerate(zip(centers, stds)):
        X_clean.append(rng.normal(loc=center, scale=std, size=(n_per_cluster, 2)))
        y_clean.extend([c] * n_per_cluster)

    X_clean = np.vstack(X_clean)
    y_clean = np.array(y_clean)

    noise = rng.uniform(low=[-7.0, -6.0], high=[7.0, 6.0], size=(60, 2))
    X = np.vstack([X_clean, noise])
    y = np.hstack([y_clean, NOISE_LABEL * np.ones(len(noise), dtype=int)])
    return X, y


def generate_sparse_legitimate(seed=42):
    """Scenario B: one legitimate but much smaller cluster (220/220/40)."""
    rng = np.random.RandomState(seed)

    centers = np.array([[-4.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
    counts = [220, 220, 40]
    stds = [0.65, 0.65, 0.65]

    X_clean, y_clean = [], []
    for c, (center, count, std) in enumerate(zip(centers, counts, stds)):
        X_clean.append(rng.normal(loc=center, scale=std, size=(count, 2)))
        y_clean.extend([c] * count)

    X_clean = np.vstack(X_clean)
    y_clean = np.array(y_clean)

    noise = rng.uniform(low=[-7.0, -6.0], high=[7.0, 6.0], size=(60, 2))
    X = np.vstack([X_clean, noise])
    y = np.hstack([y_clean, NOISE_LABEL * np.ones(len(noise), dtype=int)])
    return X, y


def generate_unequal_cluster_size(seed=42):
    """Scenario C: strongly imbalanced cluster sizes (300/120/40)."""
    rng = np.random.RandomState(seed)

    centers = np.array([[-4.0, 0.0], [0.0, 0.0], [4.0, 0.0]])
    counts = [300, 120, 40]
    stds = [0.65, 0.65, 0.65]

    X_clean, y_clean = [], []
    for c, (center, count, std) in enumerate(zip(centers, counts, stds)):
        X_clean.append(rng.normal(loc=center, scale=std, size=(count, 2)))
        y_clean.extend([c] * count)

    X_clean = np.vstack(X_clean)
    y_clean = np.array(y_clean)

    noise = rng.uniform(low=[-7.0, -6.0], high=[7.0, 6.0], size=(60, 2))
    X = np.vstack([X_clean, noise])
    y = np.hstack([y_clean, NOISE_LABEL * np.ones(len(noise), dtype=int)])
    return X, y


def generate_mixed_density_noise(seed=42):
    """Scenario D: contamination in both sparse and comparatively dense regions.

    30 peripheral low-density points plus 30 compact points around
    ``(0.0, 0.8)`` with scale 0.28, i.e. points that are locally dense and
    therefore cannot be identified by density alone.
    """
    rng = np.random.RandomState(seed)

    centers = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    n_per_cluster = 180
    std = 0.65

    X_clean, y_clean = [], []
    for c, center in enumerate(centers):
        X_clean.append(rng.normal(loc=center, scale=std, size=(n_per_cluster, 2)))
        y_clean.extend([c] * n_per_cluster)

    X_clean = np.vstack(X_clean)
    y_clean = np.array(y_clean)

    low_density_noise = rng.uniform(
        low=[-7.0, -6.0], high=[7.0, 7.0], size=(30, 2)
    )
    dense_noise = rng.normal(
        loc=np.array([0.0, 0.8]), scale=0.28, size=(30, 2)
    )
    noise = np.vstack([low_density_noise, dense_noise])

    X = np.vstack([X_clean, noise])
    y = np.hstack([y_clean, NOISE_LABEL * np.ones(len(noise), dtype=int)])
    return X, y


STRESS_SCENARIOS = {
    "Heterogeneous Density": generate_heterogeneous_density,
    "Sparse Legitimate Cluster": generate_sparse_legitimate,
    "Unequal Cluster Size": generate_unequal_cluster_size,
    "Mixed-Density Noise": generate_mixed_density_noise,
}


# ----------------------------------------------------------------------
# Contamination used by the noise-ratio sweep
# ----------------------------------------------------------------------

def add_uniform_noise(X, y, noise_ratio, rng=None):
    """Append ``floor(noise_ratio * n_clean)`` uniform points.

    The denominator convention here is the *clean* sample count, unlike the
    fixed-size benchmarks where the ratio is taken over the augmented matrix.
    """
    if rng is None:
        rng = np.random
    n_samples = X.shape[0]
    n_noise = int(n_samples * noise_ratio)
    noise = rng.uniform(X.min(axis=0), X.max(axis=0), size=(n_noise, X.shape[1]))
    X_new = np.vstack([X, noise])
    y_new = np.hstack([y, NOISE_LABEL * np.ones(n_noise)])
    return X_new, y_new
