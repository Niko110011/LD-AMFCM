"""
Experiment 1 -- sensitivity to the mapping constants alpha, beta and gamma.

Reproduces the three mapping-sensitivity analyses.  The protocol differs per
benchmark exactly as it did in the original notebook:

    Breast Cancer + Outliers   ten repetitions (init seeds 0..9)
    Moons + Gaussian Noise     single run, init seed 42
    Iris + Outliers            single run, init seed 42

In every case the contaminated matrix is generated once and frozen, and only
the initialisation seed varies between repetitions.

Only one grid constant is varied at a time; the other two stay at the
headline values alpha=6, beta=8, gamma=12.

Usage
-----
    python experiments/exp01_mapping_sensitivity.py
    python experiments/exp01_mapping_sensitivity.py --quick
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


PROTOCOL = {
    "Breast Cancer + Outliers": {"n_runs": 10, "init_seed": 0},
    "Moons + Gaussian Noise": {"n_runs": 1, "init_seed": config.SEED_SINGLE_RUN},
    "Iris + Outliers": {"n_runs": 1, "init_seed": config.SEED_SINGLE_RUN},
}

GRIDS = {
    "alpha": config.ALPHA_GRID,
    "beta": config.BETA_GRID,
    "gamma": config.GAMMA_GRID,
}

# Each grid drives the mapping that it belongs to.
MAPPING_FOR_PARAMETER = {"alpha": "Exp", "beta": "Rec", "gamma": "Sig"}

CONSTANTS = {"alpha": config.ALPHA, "beta": config.BETA, "gamma": config.GAMMA}


def run(quick=False):
    experiment = "exp01_mapping_sensitivity"
    io_utils.write_environment_report(experiment, extra={
        "protocol": "frozen input; per-benchmark data seed 42",
        "varied": "alpha, beta, gamma (one at a time)",
        "constant": CONSTANTS,
    })
    plt = io_utils.configure_matplotlib()

    records = []

    for benchmark_name, protocol in PROTOCOL.items():
        _, X_raw, y_true = datasets.get_benchmark(
            benchmark_name, data_seed=42, local_rng=True
        )
        X = datasets.standardize(X_raw)
        n_clusters = datasets.n_clusters_from_labels(y_true)

        # Density quantities do not depend on the mapping constant, so they
        # are computed once per benchmark.
        rho = density.local_density(X, k=config.K_NEIGHBORS,
                                    estimator=config.MAIN_DENSITY_ESTIMATOR)
        rho_bar = density.normalize_density(rho)
        w = density.density_weights(rho)

        n_runs = protocol["n_runs"]
        init_seeds = [protocol["init_seed"] + i for i in range(n_runs)]
        if quick:
            init_seeds = init_seeds[:2]

        print("\nbenchmark: {} | runs={}".format(benchmark_name, len(init_seeds)))

        for parameter, grid in GRIDS.items():
            grid = grid[:3] if quick else grid

            mapping_name = MAPPING_FOR_PARAMETER[parameter]

            for value in grid:
                kwargs = {"m_min": config.M_MIN, "m_max": config.M_MAX}
                kwargs[parameter] = value
                m_i = mappings.fuzzifier_values(mapping_name, rho_bar, **kwargs)

                aris = []
                for init_seed in init_seeds:
                    U = fcm.fcm_weighted(
                        X, n_clusters, m_i, w,
                        random_state=init_seed,
                        n_init=config.n_init_for("exp01_mapping_sensitivity"),
                        # The published alpha/beta/gamma curves were produced
                        # with the negative-power membership form, whose 1e-12
                        # guard dominates for crisp samples.  See the solver
                        # docstring and KNOWN_VARIATIONS.md.
                        membership_form=fcm.MEMBERSHIP_NEGATIVE_POWER,
                    )
                    aris.append(metrics.clean_ari(y_true, U.argmax(axis=1)))

                records.append({
                    "Dataset": benchmark_name,
                    "Parameter": parameter,
                    "Value": value,
                    "ARI_Mean": float(np.mean(aris)),
                    "ARI_Std": float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0,
                    "N_Runs": len(aris),
                })

            print("  {:>5s} grid done".format(parameter))

    df = pd.DataFrame(records)
    io_utils.save_table(df, experiment, "mapping_sensitivity.csv")

    # ---- figures -------------------------------------------------------
    for benchmark_name in PROTOCOL:
        subset = df[df["Dataset"] == benchmark_name]
        for parameter in GRIDS:
            part = subset[subset["Parameter"] == parameter]
            if len(part) == 0:
                continue

            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            ax.plot(part["Value"], part["ARI_Mean"], "o-",
                    linewidth=2, markersize=7, label="ARI")

            if "ARI_Std" in part and part["ARI_Std"].max() > 0:
                ax.fill_between(
                    part["Value"],
                    part["ARI_Mean"] - part["ARI_Std"],
                    part["ARI_Mean"] + part["ARI_Std"],
                    alpha=0.15,
                )

            ax.axvline(x=CONSTANTS[parameter], linestyle="--", linewidth=1.5,
                       label="headline ${}$ = {}".format(parameter,
                                                         CONSTANTS[parameter]))
            ax.set_xlabel("${}$ value".format(parameter), fontsize=12)
            ax.set_ylabel("ARI", fontsize=12)
            ax.set_title("Sensitivity of ${}$ on {}".format(parameter,
                                                            benchmark_name),
                         fontsize=13)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            fig.tight_layout()

            io_utils.save_figure(
                fig, experiment,
                "sensitivity_{}_{}.png".format(
                    parameter, benchmark_name.split()[0].lower()),
            )
            plt.close(fig)

    print("\nExperiment 1 finished.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="truncate the grids for a fast check")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
