"""
Experiment 4 -- sensitivity to the neighbourhood size k.

The neighbourhood size is the only parameter of the density estimator.  It is
varied over {5, 8, 12, 16, 20, 30} on the three benchmarks, with the mapping
constants held at their headline values.  The protocol is a single run with
initialisation seed 42 on a frozen contaminated matrix.

Note that this analysis used a single k-means++ restart, unlike the comparison
table which used ten.

By default the solver uses the ratio membership form, as the headline
configuration does.  ``--legacy-membership`` switches to the negative-power
form used by the original notebook, which reproduces its saved curve and
writes ``k_sensitivity_single_run_legacy.csv``.

Usage
-----
    python experiments/exp04_k_sensitivity.py
    python experiments/exp04_k_sensitivity.py --quick
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldamfcm import (  # noqa: E402
    config, datasets, density, fcm, io_utils, mappings, metrics,
)

BENCHMARKS = [
    ("Breast", "Breast Cancer + Outliers"),
    ("Iris", "Iris + Outliers"),
    ("Moons", "Moons + Gaussian Noise"),
]


def run(quick=False, legacy_membership=False):
    experiment = "exp04_k_sensitivity"

    membership_form = (fcm.MEMBERSHIP_NEGATIVE_POWER if legacy_membership
                       else fcm.MEMBERSHIP_RATIO)

    io_utils.write_environment_report(experiment, extra={
        "protocol": "frozen input; per-benchmark data seed 42; single run",
        "init_seed": config.SEED_SINGLE_RUN,
        "k_values": config.K_GRID,
        "kmeans_n_init": 1,
        "membership_form": membership_form,
    })
    plt = io_utils.configure_matplotlib()

    k_values = config.K_GRID[:3] if quick else config.K_GRID
    records = []

    for short_name, benchmark_name in BENCHMARKS:
        _name, X_raw, y_true = datasets.get_benchmark(
            benchmark_name, data_seed=config.SEED_SINGLE_RUN, local_rng=True
        )
        X = datasets.standardize(X_raw)
        n_clusters = datasets.n_clusters_from_labels(y_true)

        for k in k_values:
            rho = density.local_density(
                X, k=k, estimator=config.MAIN_DENSITY_ESTIMATOR
            )
            rho_bar = density.normalize_density(rho)
            w = density.density_weights(rho)

            for mapping_name in config.MAPPING_NAMES:
                m_i = mappings.fuzzifier_values(mapping_name, rho_bar)
                U = fcm.fcm_weighted(
                    X, n_clusters, m_i, w,
                    random_state=config.SEED_SINGLE_RUN,
                    n_init=config.n_init_for("exp04_k_sensitivity"),
                    membership_form=membership_form,
                )
                records.append({
                    "Dataset": short_name,
                    "k": k,
                    "Mapping": mapping_name,
                    "ARI": metrics.clean_ari(y_true, U.argmax(axis=1)),
                })

        print("  {} done".format(short_name))

    df = pd.DataFrame(records)
    filename = ("k_sensitivity_single_run_legacy.csv" if legacy_membership
                else "k_sensitivity_single_run.csv")
    io_utils.save_table(df, experiment, filename)

    # ---- one figure per dataset, one per mapping -----------------------
    for short_name, _ in BENCHMARKS:
        subset = df[df["Dataset"] == short_name]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        for mapping_name, marker in zip(config.MAPPING_NAMES, ["o", "s", "^"]):
            part = subset[subset["Mapping"] == mapping_name]
            ax.plot(part["k"], part["ARI"], marker=marker, linestyle="-",
                    linewidth=2, markersize=7, label=mapping_name)
        ax.axvline(x=config.K_NEIGHBORS, linestyle="--", linewidth=1.5,
                   label="headline $k={}$".format(config.K_NEIGHBORS))
        ax.set_xticks(k_values)
        ax.set_xlabel("$k$", fontsize=12)
        ax.set_ylabel("ARI", fontsize=12)
        ax.set_title("Sensitivity of $k$ on {}".format(short_name), fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        io_utils.save_figure(
            fig, experiment,
            "k_sensitivity_{}{}.png".format(
                short_name.lower(), "_legacy" if legacy_membership else ""))
        plt.close(fig)

    print("\nExperiment 4 finished.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--legacy-membership", action="store_true",
                        help="use the negative-power membership form used by "
                             "the original notebook; the default is the ratio "
                             "form used by the headline configuration")
    args = parser.parse_args()
    run(quick=args.quick, legacy_membership=args.legacy_membership)


if __name__ == "__main__":
    main()
