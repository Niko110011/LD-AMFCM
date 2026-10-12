"""
Experiment 8 -- ablation of the two ingredients.

LD-AMFCM combines a sample-wise fuzzifier with density-derived sample weights.
The ablation switches each ingredient on and off separately:

    FCM                  constant fuzzifier,  uniform weights
    FCM + w_i            constant fuzzifier,  density weights
    FCM + m_i            sample-wise fuzzifier (sigmoid),  uniform weights
    LD-AMFCM (w_i + m_i) sample-wise fuzzifier (sigmoid),  density weights

on the five benchmarks, with MC-FCM included as an external adaptive
fuzzifier.  Single run, initialisation seed 42.

The benchmarks are generated in the declared order from one shared stream
seeded once at 42, which is what the original study did.

Usage
-----
    python experiments/exp08_ablation.py
    python experiments/exp08_ablation.py --quick
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldamfcm import (  # noqa: E402
    config, datasets, density, fcm, io_utils, mappings, metrics, models,
)

VARIANTS = [
    ("FCM", "constant_m", "uniform_w"),
    ("FCM + w_i", "constant_m", "density_w"),
    ("FCM + m_i", "adaptive_m_sigmoid", "uniform_w"),
    ("LD-AMFCM (w_i + m_i)", "adaptive_m_sigmoid", "density_w"),
    ("MC-FCM", "mc_fcm_m", "uniform_w"),
]

ADAPTIVE_MAPPING = "Sig"


def run(quick=False):
    experiment = "exp08_ablation"
    init_seed = config.SEED_SINGLE_RUN

    io_utils.write_environment_report(experiment, extra={
        "protocol": "frozen input; shared stream seeded once at 42",
        "init_seed": init_seed,
        "kmeans_n_init": config.KMEANS_N_INIT,
        "adaptive_mapping": ADAPTIVE_MAPPING,
        "variants": [name for name, _, _ in VARIANTS],
    })
    plt = io_utils.configure_matplotlib()

    benchmarks = datasets.get_benchmarks(data_seed=init_seed, local_rng=False)
    if quick:
        benchmarks = benchmarks[:1]

    records = []

    for benchmark_name, X_raw, y_true in benchmarks:
        X = datasets.standardize(X_raw)
        n_clusters = datasets.n_clusters_from_labels(y_true)
        n = X.shape[0]

        rho = density.local_density(X, k=config.K_NEIGHBORS,
                                    estimator=config.MAIN_DENSITY_ESTIMATOR)
        rho_bar = density.normalize_density(rho)
        w_density = density.density_weights(rho)
        w_uniform = np.ones(n)

        fuzzifiers = {
            "constant_m": np.ones(n) * 2.0,
            "adaptive_m_sigmoid": mappings.fuzzifier_values(ADAPTIVE_MAPPING,
                                                            rho_bar),
            "mc_fcm_m": models.mc_fcm_fuzzifier(X, n_clusters),
        }
        weights = {"uniform_w": w_uniform, "density_w": w_density}

        print("\nbenchmark: {}".format(benchmark_name))

        for variant_name, fuzzifier_key, weight_key in VARIANTS:
            m_i = fuzzifiers[fuzzifier_key]
            w_i = weights[weight_key]

            U = fcm.fcm_weighted(X, n_clusters, m_i, w_i,
                                 random_state=init_seed,
                                 n_init=config.KMEANS_N_INIT)
            labels = U.argmax(axis=1)

            records.append({
                "Dataset": benchmark_name,
                "Method": variant_name,
                "ARI": metrics.clean_ari(y_true, labels),
                "DBI": metrics.clean_dbi(X, y_true, labels),
                "Noise_Membership": metrics.noise_membership(U, y_true),
                "Noise_AUC": metrics.noise_auc(U, y_true),
            })

        print("  {} variants done".format(len(VARIANTS)))

    df = pd.DataFrame(records)
    io_utils.save_table(df, experiment, "ablation_results.csv")

    # Overall summary across benchmarks, matching the reported aggregate.
    overall = df.groupby("Method").agg({
        "ARI": ["mean", "std"],
        "DBI": ["mean", "std"],
        "Noise_Membership": ["mean", "std"],
        "Noise_AUC": ["mean", "std"],
    })
    io_utils.save_table(overall, experiment, "ablation_overall_summary.csv",
                        index=True)

    print("\nExperiment 8 finished.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
