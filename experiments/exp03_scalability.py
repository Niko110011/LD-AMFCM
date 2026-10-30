"""
Experiment 3 -- scalability in sample size and dimensionality.

Synthetic data with three clusters.  Only the first two features carry cluster
structure; the remaining ``p - 2`` features are pure noise dimensions, so the
density estimate is computed in a space whose informative content does not
grow with ``p``.

The data are generated once per configuration with a fixed generator seed and
are therefore common to all five repetitions; only the initialisation seed
varies.  Three timings are separated and reported independently because they
scale differently:

    density estimation   kNN search, depends on n and p
    initialisation       k-means++ prototypes
    optimisation         the alternating fuzzy updates

Note on the convergence criterion: this study used the largest per-cluster
prototype shift, whereas every other experiment used the Frobenius norm of the
prototype change.  Both are applied at the same tolerance of 1e-4.

Usage
-----
    python experiments/exp03_scalability.py
    python experiments/exp03_scalability.py --quick
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldamfcm import (  # noqa: E402
    config, datasets, density, fcm, io_utils, mappings,
)

METHODS = [
    config.METHOD_FCM,
    config.METHOD_LD_EXP,
    config.METHOD_LD_REC,
    config.METHOD_LD_SIG,
]

# The scalability study ran the kNN search with the brute-force algorithm and
# all cores.
#
# WARNING -- this default is retained only to reproduce the published timing
# columns.  In this environment the parallel path is dominated by worker-pool
# start-up, not by computation: for n=500 the call takes ~3.95 s with
# n_jobs=-1 against ~0.005 s with n_jobs=1, and even at n=10000 it is ~5.54 s
# against ~1.72 s.  Timing figures produced with n_jobs=-1 therefore measure
# thread start-up, not the estimator.  Run with --n-jobs 1 for timings that
# reflect the algorithm.  See docs/KNOWN_VARIATIONS.md, item 14.
DENSITY_ALGORITHM = "brute"
DENSITY_N_JOBS = -1


def run_one_configuration(X, n_clusters, init_seed, n_jobs=DENSITY_N_JOBS,
                          membership_form=fcm.MEMBERSHIP_RATIO):
    """Time FCM and the three LD-AMFCM mappings on one configuration."""
    results = []

    U, _V, n_iter, init_time, opt_time = fcm.fcm_weighted(
        X, n_clusters, np.ones(len(X)) * 2.0, np.ones(len(X)),
        random_state=init_seed,
        n_init=config.n_init_for("exp03_scalability"),
        membership_form=membership_form,
        convergence=fcm.PROTOTYPE_MAXSHIFT, return_diagnostics=True,
    )
    results.append({
        "Method": config.METHOD_FCM,
        "Density_Time": 0.0,
        "Initialization_Time": init_time,
        "Optimization_Time": opt_time,
        "Total_Time": init_time + opt_time,
        "Iterations": n_iter,
    })

    rho, density_time = density.measure_density_time(
        X, k=config.K_NEIGHBORS, estimator=config.MAIN_DENSITY_ESTIMATOR,
        algorithm=DENSITY_ALGORITHM, n_jobs=n_jobs,
    )
    rho_bar = density.normalize_density(rho)
    w = density.density_weights(rho)

    for mapping_name in config.MAPPING_NAMES:
        m_i = mappings.fuzzifier_values(mapping_name, rho_bar)
        U, _V, n_iter, init_time, opt_time = fcm.fcm_weighted(
            X, n_clusters, m_i, w,
            random_state=init_seed,
            n_init=config.n_init_for("exp03_scalability"),
            membership_form=membership_form,
            convergence=fcm.PROTOTYPE_MAXSHIFT, return_diagnostics=True,
        )
        results.append({
            "Method": config.mapping_label(mapping_name),
            "Density_Time": density_time,
            "Initialization_Time": init_time,
            "Optimization_Time": opt_time,
            "Total_Time": density_time + init_time + opt_time,
            "Iterations": n_iter,
        })

    return results


def sweep(experiment, axis, values, seeds, n_jobs=DENSITY_N_JOBS,
          membership_form=fcm.MEMBERSHIP_RATIO, suffix=""):
    """Run the sweep over either the sample size or the dimensionality.

    ``axis="n"`` varies the sample count at fixed ``p = FIXED_P``;
    ``axis="p"`` varies the dimensionality at fixed ``n = FIXED_N``.
    """
    records = []
    n_clusters = config.N_CLUSTERS_SCALABILITY

    for value in values:
        n = int(value) if axis == "n" else int(config.FIXED_N)
        p = int(config.FIXED_P) if axis == "n" else int(value)

        print("  {} = {:<6d} (other dimension {})".format(
            axis, n if axis == "n" else p, p if axis == "n" else n))

        for init_seed in seeds:
            X_raw, _y = datasets.generate_scalability_data(
                n=n, p=p, n_clusters=n_clusters,
                seed=config.SCALABILITY_GENERATOR_SEED,
            )
            X = datasets.standardize(X_raw)

            for row in run_one_configuration(
                    X, n_clusters, init_seed, n_jobs=n_jobs,
                    membership_form=membership_form):
                row.update({"n": n, "p": p, "Seed": init_seed, "Axis": axis})
                records.append(row)

    raw = pd.DataFrame(records)
    io_utils.save_table(raw, experiment,
                        "scalability_{}_raw{}.csv".format(axis, suffix))

    summary = (
        raw.groupby(["Method", "n", "p"])
        .agg({"Density_Time": ["mean", "std"],
              "Initialization_Time": ["mean", "std"],
              "Optimization_Time": ["mean", "std"],
              "Total_Time": ["mean", "std"],
              "Iterations": ["mean", "std"]})
        .reset_index()
    )
    summary.columns = [
        "_".join(str(x) for x in col).strip("_") for col in summary.columns
    ]
    io_utils.save_table(summary, experiment,
                        "scalability_{}_summary{}.csv".format(axis, suffix))
    return summary


def plot_axis(summary, axis, experiment, suffix=""):
    plt = io_utils.configure_matplotlib()
    grid_key = axis if axis in ("n", "p") else "n"

    fig, ax = plt.subplots(figsize=(9, 6))
    for method in METHODS:
        part = summary[summary["Method"] == method].sort_values(grid_key)
        ax.plot(part[grid_key], part["Total_Time_mean"], marker="o",
                linewidth=2, label=method)
    ax.set_xlabel("Number of samples (n)" if grid_key == "n"
                  else "Feature dimension (p)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Total runtime vs {}".format(grid_key))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    io_utils.save_figure(fig, experiment,
                         "scalability_{}_total_time{}.png".format(axis, suffix))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for mapping_name in config.MAPPING_NAMES:
        method = config.mapping_label(mapping_name)
        part = summary[summary["Method"] == method].sort_values(grid_key)
        ax.plot(part[grid_key], part["Density_Time_mean"], marker="o",
                linewidth=2, label=method)
    ax.set_xlabel("Number of samples (n)" if grid_key == "n"
                  else "Feature dimension (p)")
    ax.set_ylabel("kNN density estimation time (s)")
    ax.set_title("Density estimation cost vs {}".format(grid_key))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    io_utils.save_figure(fig, experiment,
                         "scalability_{}_density_time{}.png".format(
                             axis, suffix))
    plt.close(fig)


def run(quick=False, n_jobs=DENSITY_N_JOBS, legacy_membership=False):
    experiment = "exp03_scalability"
    seeds = list(range(2 if quick else config.N_REPEATS_SCALABILITY))

    membership_form = (fcm.MEMBERSHIP_NEGATIVE_POWER if legacy_membership
                       else fcm.MEMBERSHIP_RATIO)
    suffix = "_legacy" if legacy_membership else ""

    io_utils.write_environment_report(experiment, extra={
        "protocol": "frozen synthetic data, generator seed {}".format(
            config.SCALABILITY_GENERATOR_SEED),
        "init_seeds": seeds,
        "n_values": config.N_VALUES,
        "p_values": config.P_VALUES,
        "fixed_p": config.FIXED_P,
        "fixed_n": config.FIXED_N,
        "density_algorithm": DENSITY_ALGORITHM,
        "density_n_jobs": n_jobs,
        "membership_form": membership_form,
        "n_init": config.n_init_for("exp03_scalability"),
        "timing": "time.perf_counter, wall-clock seconds",
        "caveat": ("n_jobs=-1 measures worker-pool start-up in this "
                   "environment; use --n-jobs 1 for algorithmic timings"
                   if n_jobs == -1 else "sequential kNN search"),
    }, suffix=suffix)

    n_values = config.N_VALUES[:2] if quick else config.N_VALUES
    p_values = config.P_VALUES[:2] if quick else config.P_VALUES

    started = time.perf_counter()

    print("\nsample-size sweep")
    summary_n = sweep(experiment, "n", n_values, seeds, n_jobs=n_jobs,
                      membership_form=membership_form, suffix=suffix)
    plot_axis(summary_n, "n", experiment, suffix=suffix)

    print("\ndimensionality sweep")
    summary_p = sweep(experiment, "p", p_values, seeds, n_jobs=n_jobs,
                      membership_form=membership_form, suffix=suffix)
    plot_axis(summary_p, "p", experiment, suffix=suffix)

    print("\nExperiment 3 finished in {:.2f} s.".format(
        time.perf_counter() - started))
    return summary_n, summary_p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=DENSITY_N_JOBS,
                        help="parallelism of the exact kNN search. The default "
                             "of -1 reproduces the published figure but "
                             "measures worker-pool start-up; use 1 for "
                             "timings that reflect the algorithm")
    parser.add_argument("--legacy-membership", action="store_true",
                        help="use the negative-power membership form used by "
                             "the original notebook; the default is the ratio "
                             "form used by the headline configuration")
    args = parser.parse_args()
    run(quick=args.quick, n_jobs=args.n_jobs,
        legacy_membership=args.legacy_membership)


if __name__ == "__main__":
    main()
