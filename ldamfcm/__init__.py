"""
LD-AMFCM reproducibility package.

Local-density adaptive fuzzy c-means, as reported in the manuscript.
The package reproduces the experiments of the experimental notebook:
the parameter sensitivity analyses, the noise stress tests, the scalability
study, the comparison against published baselines, the ablation, the
density-reliability stress scenarios and the statistical protocol.

Quick start
-----------
    from ldamfcm import datasets, density, mappings, fcm, metrics

    benchmarks = datasets.get_benchmarks(data_seed=42)
    name, X_raw, y_true = benchmarks[0]
    X = datasets.standardize(X_raw)

    rho = density.local_density(X, k=12)
    rho_bar = density.normalize_density(rho)
    w = density.density_weights(rho)
    m_i = mappings.fuzzifier_values("Exp", rho_bar)

    U = fcm.fcm_weighted(X, datasets.n_clusters_from_labels(y_true), m_i, w,
                         random_state=0, n_init=10)
    print(metrics.clean_ari(y_true, U.argmax(axis=1)))
"""

__version__ = "1.0.0"

__all__ = [
    "config",
    "datasets",
    "density",
    "fcm",
    "io_utils",
    "mappings",
    "metrics",
    "models",
    "stats",
]
