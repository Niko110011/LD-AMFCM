"""
Experiment 11 -- repeated-run comparison and significance testing.

This is the *regenerated-input* protocol: every repetition regenerates all
five contaminated matrices and also re-initialises every method.  It is
reported separately from the frozen-input comparison and the two are never
pooled, because in this protocol the difference between two methods on the
same run index reflects both the data and the initialisation.

Faithful mode (default)
-----------------------
The original study seeded the global NumPy generator once per repetition and
then let the k-means++ initialisation draw from that same global stream, so
the initial prototype of a method depends on how much randomness the earlier
steps of that run consumed.  The package reproduces this exactly: the
prototypes are initialised with ``random_state=None`` after the global
generator has been seeded and used for the data.

Clean mode (``--explicit-init-seed``)
-------------------------------------
Pass the run index as an explicit initialisation seed instead.  This removes
the reliance on the global stream, at the cost of no longer reproducing the
published statistics exactly.  It is the form to use if the experiment is ever
re-run for publication.

Statistics
----------
    paired t-test vs the FCM baseline on matched runs
    Cohen's d_z for paired samples
    Holm step-down correction over all dataset x method comparisons
    Friedman omnibus test on the dataset-level mean ARI, with Kendall's W
    Wilcoxon signed-rank post-hoc tests against FCM, Holm corrected

Usage
-----
    python experiments/exp11_paired_statistics.py
    python experiments/exp11_paired_statistics.py --explicit-init-seed
    python experiments/exp11_paired_statistics.py --quick
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldamfcm import (  # noqa: E402
    config, datasets, density, fcm, io_utils, mappings, metrics, pipeline, stats,
)

METHODS = [
    config.METHOD_FCM,
    config.METHOD_DW_FCM,
    config.METHOD_LD_EXP,
    config.METHOD_LD_REC,
    config.METHOD_LD_SIG,
]


def run(quick=False, explicit_init_seed=False):
    experiment = "exp11_paired_statistics"
    seed_spec = config.SEEDS_REGENERATED_INPUT

    n_runs = 2 if quick else seed_spec.n_runs

    io_utils.write_environment_report(experiment, extra={
        "protocol": "regenerated input",
        "data_seed": "s, s = 0..{}".format(n_runs - 1),
        "initialisation": ("explicit random_state = s" if explicit_init_seed
                           else "global stream after np.random.seed(s)"),
        "n_runs": n_runs,
        "benchmark_order": config.BENCHMARK_ORDER,
    })
    plt = io_utils.configure_matplotlib()

    records = []

    for seed in range(n_runs):
        np.random.seed(seed)
        benchmarks = datasets.get_benchmarks(data_seed=seed, local_rng=False)
        if quick:
            benchmarks = benchmarks[:2]

        for benchmark_name, X_raw, y_true in benchmarks:
            X = datasets.standardize(X_raw)
            n_clusters = datasets.n_clusters_from_labels(y_true)

            rho = density.local_density(
                X, k=config.K_NEIGHBORS, estimator=config.MAIN_DENSITY_ESTIMATOR
            )
            rho_bar = density.normalize_density(rho)
            w = density.density_weights(rho)

            for method in METHODS:
                m_i, w_i = pipeline.method_vectors(
                    method, X, n_clusters, rho_bar, w
                )
                random_state = seed if explicit_init_seed else None
                U = fcm.fcm_weighted(X, n_clusters, m_i, w_i,
                                     random_state=random_state,
                                     n_init=config.KMEANS_N_INIT)
                labels = U.argmax(axis=1)

                records.append({
                    "Run": seed,
                    "Dataset": benchmark_name,
                    "Method": method,
                    "ARI": metrics.clean_ari(y_true, labels),
                    "DBI": metrics.clean_dbi(X, y_true, labels),
                    "Noise_Membership": metrics.noise_membership(U, y_true),
                })

        print("  run {} / {} done".format(seed + 1, n_runs))

    df = pd.DataFrame(records)
    io_utils.save_table(df, experiment, "paired_runs.csv")

    # ---- mean +/- std per dataset and method ---------------------------
    summary = df.groupby(["Dataset", "Method"]).agg({
        "ARI": ["mean", "std"],
        "DBI": ["mean", "std"],
        "Noise_Membership": ["mean", "std"],
    }).reset_index()
    display = pd.DataFrame({
        "Dataset": summary["Dataset"],
        "Method": summary["Method"],
        "ARI": [io_utils.mean_std_str(m, s) for m, s in
                zip(summary[("ARI", "mean")], summary[("ARI", "std")])],
        "DBI": [io_utils.mean_std_str(m, s) for m, s in
                zip(summary[("DBI", "mean")], summary[("DBI", "std")])],
        "Noise_Membership": [
            io_utils.mean_std_str(m, s) for m, s in
            zip(summary[("Noise_Membership", "mean")],
                summary[("Noise_Membership", "std")])
        ],
    })
    io_utils.save_table(display, experiment, "mean_std_summary.csv")

    # ---- paired t-tests with Cohen's d_z and Holm correction -----------
    paired = stats.paired_tests_vs_baseline(records, config.METHOD_FCM)
    io_utils.save_table(paired, experiment, "paired_ttest_holm.csv")

    print("\npaired t-test vs FCM (Holm corrected)")
    print(paired.to_string(index=False))

    # ---- Friedman and Wilcoxon on dataset-level means ------------------
    mean_ari = (
        df.groupby(["Dataset", "Method"])["ARI"].mean().reset_index()
    )
    friedman_table = mean_ari.pivot(index="Dataset", columns="Method",
                                    values="ARI")[METHODS]

    friedman_stat, friedman_p, kendall_w = stats.friedman_test(
        friedman_table, METHODS
    )

    friedman_row = pd.DataFrame([{
        "Statistic": friedman_stat,
        "p-value": friedman_p,
        "Kendall_W": kendall_w,
        "N_Datasets": len(friedman_table),
        "N_Methods": len(METHODS),
    }])
    io_utils.save_table(friedman_row, experiment, "friedman_test.csv")

    print("\nFriedman: statistic={:.5f}, p={:.5f}, Kendall's W={:.5f}".format(
        friedman_stat, friedman_p, kendall_w))

    posthoc = stats.wilcoxon_posthoc(
        friedman_table, METHODS, config.METHOD_FCM
    )
    io_utils.save_table(posthoc, experiment, "wilcoxon_posthoc.csv")

    print("\nWilcoxon post-hoc vs FCM (Holm corrected)")
    print(posthoc.to_string(index=False))

    io_utils.save_table(friedman_table.reset_index(), experiment,
                        "dataset_level_mean_ari.csv")

    print("\nExperiment 11 finished.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--explicit-init-seed", action="store_true",
                        help="pass the run index as an explicit init seed "
                             "instead of relying on the global stream")
    args = parser.parse_args()
    run(quick=args.quick, explicit_init_seed=args.explicit_init_seed)


if __name__ == "__main__":
    main()
