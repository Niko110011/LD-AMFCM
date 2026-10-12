"""
Quick verification that the package reproduces the published numbers.

Runs the frozen-input comparison protocol on a single benchmark for the ten
initialisation seeds and compares the resulting ARI against the values saved
in the original notebook output.  This takes a couple of minutes and is the
cheapest way to check that the environment and the code agree with the paper.

    python smoke_test.py                 # Breast benchmark (default)
    python smoke_test.py --benchmark Iris
"""

import argparse

import numpy as np

from ldamfcm import config, datasets, density, fcm, mappings, metrics, models


# ARI mean over the ten initialisation seeds, as saved in the original
# notebook output for the frozen-input protocol.
PUBLISHED_ARI = {
    "Blobs + Uniform Noise": {
        "FCM (Baseline)": 0.9347, "DW-FCM": 0.9347,
        "LD-AMFCM (Exp)": 0.9417, "LD-AMFCM (Rec)": 0.9347,
        "LD-AMFCM (Sig)": 0.9417, "MC-FCM": 0.9417,
    },
    "Iris + Outliers": {
        "FCM (Baseline)": 0.5638, "DW-FCM": 0.8015,
        "LD-AMFCM (Exp)": 0.8017, "LD-AMFCM (Rec)": 0.8015,
        "LD-AMFCM (Sig)": 0.8340, "MC-FCM": 0.7184,
    },
    "Moons + Gaussian Noise": {
        "FCM (Baseline)": 0.3499, "DW-FCM": 0.3659,
        "LD-AMFCM (Exp)": 0.3741, "LD-AMFCM (Rec)": 0.3659,
        "LD-AMFCM (Sig)": 0.3907, "MC-FCM": 0.3499,
    },
    "Anisotropic + Noise": {
        "FCM (Baseline)": 0.5179, "DW-FCM": 0.5942,
        "LD-AMFCM (Exp)": 0.5850, "LD-AMFCM (Rec)": 0.5997,
        "LD-AMFCM (Sig)": 0.6144, "MC-FCM": 0.5179,
    },
    "Breast Cancer + Outliers": {
        "FCM (Baseline)": 0.6704, "DW-FCM": 0.6442,
        "LD-AMFCM (Exp)": 0.7432, "LD-AMFCM (Rec)": 0.7313,
        "LD-AMFCM (Sig)": 0.6163, "MC-FCM": 0.7062,
    },
}

TOLERANCE = 5e-5


def run_benchmark(benchmark_name):
    seed_spec = config.SEEDS_MAIN_COMPARISON

    # The contaminated matrix is generated once and then frozen; only the
    # initialisation seed varies across the repetitions.
    benchmarks = datasets.get_benchmarks(data_seed=seed_spec.data_seed,
                                         local_rng=False)
    lookup = {name: (X_raw, y) for name, X_raw, y in benchmarks}
    X_raw, y_true = lookup[benchmark_name]

    X = datasets.standardize(X_raw)
    n_clusters = datasets.n_clusters_from_labels(y_true)

    rho = density.local_density(X, k=config.K_NEIGHBORS,
                                estimator=config.MAIN_DENSITY_ESTIMATOR)
    rho_bar = density.normalize_density(rho)
    w = density.density_weights(rho)

    method_specs = [
        (config.METHOD_FCM, np.ones(len(X)) * 2.0, np.ones(len(X))),
        (config.METHOD_DW_FCM, np.ones(len(X)) * 2.0, w),
    ]
    for mapping_name in config.MAPPING_NAMES:
        m_i = mappings.fuzzifier_values(mapping_name, rho_bar)
        method_specs.append((config.mapping_label(mapping_name), m_i, w))

    method_specs.append((
        config.METHOD_MC_FCM,
        models.mc_fcm_fuzzifier(X, n_clusters),
        np.ones(len(X)),
    ))

    print("benchmark : {}  (n={}, p={}, C={})".format(
        benchmark_name, X.shape[0], X.shape[1], n_clusters))
    print("protocol  : frozen input, data seed {}, init seeds 0..9".format(
        seed_spec.data_seed))
    print("")
    print("{:<18} {:>12} {:>12} {:>10}".format(
        "method", "reproduced", "published", "delta"))

    ok = True
    for method_name, m_i, w_i in method_specs:
        aris = []
        for init_seed in seed_spec.init_seeds():
            U = fcm.fcm_weighted(
                X, n_clusters, m_i, w_i,
                random_state=init_seed,
                n_init=config.KMEANS_N_INIT,
            )
            aris.append(metrics.clean_ari(y_true, U.argmax(axis=1)))

        reproduced = float(np.mean(aris))
        published = PUBLISHED_ARI[benchmark_name][method_name]
        delta = reproduced - published
        flag = "ok" if abs(delta) < TOLERANCE else "MISMATCH"
        if abs(delta) >= TOLERANCE:
            ok = False

        print("{:<18} {:>12.4f} {:>12.4f} {:>10.4f}  {}".format(
            method_name, reproduced, published, delta, flag))

    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default="Breast Cancer + Outliers",
        choices=config.BENCHMARK_ORDER,
        help="benchmark to verify",
    )
    args = parser.parse_args()

    ok = run_benchmark(args.benchmark)
    print("")
    print("RESULT: {}".format("reproduced" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
