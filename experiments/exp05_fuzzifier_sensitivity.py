"""
Experiment 5 -- sensitivity to the fuzzifier bounds m_min and m_max.

By default this experiment follows the density estimator defined in the
manuscript,

    rho_i = (1/k) * sum_j exp(-||x_i - x_j||^2 / sigma^2),
    sigma = max(global median of the retained kNN distances, 1e-12),

and the same solver as the headline configuration.  A change in m_min or m_max
is therefore the only difference from the default point, and the default
point agrees with the k sweep.  The output file is
``fuzzifier_sensitivity_paper.csv`` and every row carries
``Configuration = paper``.

The original notebook produced these two tables with a different
implementation: a per-point bandwidth, a sum of kernels instead of a mean, a
random-membership initialisation and a membership-change stopping rule.  That
implementation does not match the manuscript definition, and its default point
disagrees with the k sweep (Iris, exponential mapping: 0.5638 against 0.8176).
It is retained behind ``--legacy-variant`` for historical reproduction only;
its output is ``fuzzifier_sensitivity_legacy_variant.csv``.

Grids
-----
    m_min in {1.02, 1.05, 1.10, 1.20, 1.30, 1.50}
    m_max in {2.0, 2.5, 3.0, 3.5, 4.0, 5.0}

with the other bound held at its default value, on Breast, Iris and Moons,
single run, initialisation seed 42.

Usage
-----
    python experiments/exp05_fuzzifier_sensitivity.py
    python experiments/exp05_fuzzifier_sensitivity.py --legacy-variant
    python experiments/exp05_fuzzifier_sensitivity.py --quick
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

GRIDS = {"m_min": config.M_MIN_GRID, "m_max": config.M_MAX_GRID}


def run(quick=False, legacy_variant=False):
    """Run the fuzzifier-bound sweep.

    ``legacy_variant`` switches to the implementation the original notebook
    used for these two tables.  That implementation does not match the
    density estimator defined in the manuscript (per-point bandwidth and a sum
    of kernels rather than a global median and a mean), so its default point
    disagrees with the ``k`` sweep.  It is retained only for historical
    reproduction; the default here follows the manuscript.
    """
    experiment = "exp05_fuzzifier_sensitivity"

    estimator = (config.VARIANT_DENSITY_ESTIMATOR if legacy_variant
                 else config.MAIN_DENSITY_ESTIMATOR)
    configuration = "legacy_variant" if legacy_variant else "paper"
    init_seed = config.SEED_SINGLE_RUN

    io_utils.write_environment_report(experiment, extra={
        "protocol": "frozen input; per-benchmark data seed 42; single run",
        "configuration": configuration,
        "density_estimator": estimator,
        "density_definition": (
            "rho = (1/k) sum_j exp(-d^2 / sigma^2), "
            "sigma = max(global median kNN distance, 1e-12)"
            if not legacy_variant else
            "legacy: rho = sum_j exp(-d^2 / (2 sigma_i^2)), "
            "sigma_i = mean kNN distance of point i"),
        "initialisation": ("k-means++" if not legacy_variant
                           else "random membership, RandomState(42)"),
        "n_init": (config.n_init_for("exp05_fuzzifier_sensitivity")
                   if not legacy_variant else "not used"),
        "membership_form": ("negative_power (legacy random-membership solver)"
                            if legacy_variant else "ratio"),
        "stopping_rule": ("|V_new - V| < 1e-4" if not legacy_variant
                          else "|U_new - U| < 1e-5"),
        "m_min_grid": config.M_MIN_GRID,
        "m_max_grid": config.M_MAX_GRID,
    }, suffix="_legacy_variant" if legacy_variant else "")
    plt = io_utils.configure_matplotlib()

    records = []

    for short_name, benchmark_name in BENCHMARKS:
        _name, X_raw, y_true = datasets.get_benchmark(
            benchmark_name, data_seed=42, local_rng=True
        )
        X = datasets.standardize(X_raw)
        n_clusters = datasets.n_clusters_from_labels(y_true)

        rho = density.local_density(X, k=config.K_NEIGHBORS, estimator=estimator)
        rho_bar = density.normalize_density(rho)
        w = density.density_weights(rho)

        for parameter, grid in GRIDS.items():
            grid = grid[:3] if quick else grid

            for value in grid:
                kwargs = {"m_min": config.M_MIN, "m_max": config.M_MAX}
                kwargs[parameter] = value

                for mapping_name in config.MAPPING_NAMES:
                    m_i = mappings.fuzzifier_values(mapping_name, rho_bar,
                                                    **kwargs)

                    if legacy_variant:
                        U = fcm.fcm_weighted_random_membership(
                            X, n_clusters, m_i, w,
                            random_state=init_seed,
                            tol=config.VARIANT_TOL,
                        )
                    else:
                        # The manuscript's method: the same solver as the
                        # headline configuration, so a change in m_min or
                        # m_max is the only difference from the default point.
                        U = fcm.fcm_weighted(
                            X, n_clusters, m_i, w,
                            random_state=init_seed,
                            n_init=config.n_init_for(
                                "exp05_fuzzifier_sensitivity"),
                        )

                    records.append({
                        "Dataset": short_name,
                        "Parameter": parameter,
                        "Value": value,
                        "Mapping": mapping_name,
                        "ARI": metrics.clean_ari(y_true, U.argmax(axis=1)),
                        "Configuration": configuration,
                    })

        print("  {} done ({})".format(short_name, configuration))

    df = pd.DataFrame(records)
    io_utils.save_table(
        df, experiment,
        "fuzzifier_sensitivity_{}.csv".format(configuration),
    )

    # ---- figures -------------------------------------------------------
    for short_name, _ in BENCHMARKS:
        for parameter in GRIDS:
            part = df[(df["Dataset"] == short_name) &
                      (df["Parameter"] == parameter)]
            if len(part) == 0:
                continue

            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            for mapping_name, marker in zip(config.MAPPING_NAMES, ["o", "s", "^"]):
                sub = part[part["Mapping"] == mapping_name].sort_values("Value")
                ax.plot(sub["Value"], sub["ARI"], marker=marker, linestyle="-",
                        linewidth=2, markersize=7, label=mapping_name)

            headline = config.M_MIN if parameter == "m_min" else config.M_MAX
            ax.axvline(x=headline, linestyle="--", linewidth=1.5,
                       label="headline ${}$ = {}".format(parameter, headline))
            ax.set_xlabel("${}$".format(parameter), fontsize=12)
            ax.set_ylabel("ARI", fontsize=12)
            ax.set_title("Sensitivity of ${}$ on {} ({})".format(
                parameter, short_name, configuration), fontsize=13)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            io_utils.save_figure(
                fig, experiment,
                "fuzzifier_{}_{}_{}.png".format(
                    parameter, short_name.lower(), configuration),
            )
            plt.close(fig)

    print("\nExperiment 5 finished (configuration: {}).".format(configuration))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--legacy-variant", action="store_true",
                        help="reproduce the original notebook's tables, which "
                             "used a density estimator that does not match the "
                             "manuscript (per-point bandwidth, sum of kernels)")
    args = parser.parse_args()
    run(quick=args.quick, legacy_variant=args.legacy_variant)


if __name__ == "__main__":
    main()
