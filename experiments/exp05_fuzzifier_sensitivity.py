"""
Experiment 5 -- sensitivity to the fuzzifier bounds m_min and m_max.

.. warning::

   This experiment uses the **solver variant**.  It is not the headline
   configuration.  Reproducing it as it was run therefore means using the
   variant's density estimator, its random membership initialisation and its
   membership-change stopping rule.  Every table written here carries a
   ``Configuration`` column that says ``variant``.  See
   ``docs/KNOWN_VARIATIONS.md``.

Grids
-----
    m_min in {1.02, 1.05, 1.10, 1.20, 1.30, 1.50}
    m_max in {2.0, 2.5, 3.0, 3.5, 4.0, 5.0}

with the other bound held at its headline value, on Breast, Iris and Moons,
single run, initialisation seed 42.

Use ``--headline-solver`` to run the same grids with the headline solver and
the headline density estimator instead.  That produces different numbers on
purpose: it answers the question "what would this sweep have looked like under
the configuration the paper actually proposes", and is the variant to report
if the sweep is ever re-run for publication.

Usage
-----
    python experiments/exp05_fuzzifier_sensitivity.py
    python experiments/exp05_fuzzifier_sensitivity.py --headline-solver
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


def run(quick=False, headline_solver=False):
    experiment = "exp05_fuzzifier_sensitivity"

    estimator = (config.MAIN_DENSITY_ESTIMATOR if headline_solver
                 else config.VARIANT_DENSITY_ESTIMATOR)
    configuration = "headline" if headline_solver else "variant"
    init_seed = config.SEED_SINGLE_RUN

    io_utils.write_environment_report(experiment, extra={
        "protocol": "frozen input; per-benchmark data seed 42; single run",
        "configuration": configuration,
        "density_estimator": estimator,
        "initialisation": ("k-means++" if headline_solver
                           else "random membership, RandomState(42)"),
        "stopping_rule": ("|V_new - V| < 1e-4" if headline_solver
                          else "|U_new - U| < 1e-5"),
        "m_min_grid": config.M_MIN_GRID,
        "m_max_grid": config.M_MAX_GRID,
    })
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

                    if headline_solver:
                        U = fcm.fcm_weighted(
                            X, n_clusters, m_i, w,
                            random_state=init_seed,
                            n_init=config.n_init_for(
                                "exp05_fuzzifier_sensitivity"),
                            # The sweep family uses the negative-power form,
                            # so its headline counterpart does too.
                            membership_form=fcm.MEMBERSHIP_NEGATIVE_POWER,
                        )
                    else:
                        U = fcm.fcm_weighted_random_membership(
                            X, n_clusters, m_i, w,
                            random_state=init_seed,
                            tol=config.VARIANT_TOL,
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
    parser.add_argument("--headline-solver", action="store_true",
                        help="use the headline solver instead of the variant")
    args = parser.parse_args()
    run(quick=args.quick, headline_solver=args.headline_solver)


if __name__ == "__main__":
    main()
