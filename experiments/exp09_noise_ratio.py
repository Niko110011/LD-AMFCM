"""
Experiment 9 -- contamination-ratio sensitivity.

Uniform contamination is injected at 10 %, 15 % and 20 % on Iris and Breast
Cancer, and only the two end points of the method family are compared:
plain FCM and LD-AMFCM with the exponential mapping.

Denominator convention
----------------------
The number of injected points is ``floor(ratio * n_clean)``, so the ratio here
is taken over the *clean* sample count.  This is a different convention from
the fixed-size benchmarks, where the ratio is taken over the augmented matrix.
The two are not interchangeable.

Reproducibility note
--------------------
In the original notebook this experiment drew its contamination without
resetting the generator, so the exact points depended on the interpreter's
global state at the time the cell was executed.  This script resets the
generator explicitly with ``np.random.seed(42)`` before the sweep, which makes
the experiment reproducible.  The published numbers are reproduced only if
the pointer is in the same position; see docs/KNOWN_VARIATIONS.md.

Usage
-----
    python experiments/exp09_noise_ratio.py
    python experiments/exp09_noise_ratio.py --quick
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldamfcm import (  # noqa: E402
    config, datasets, density, fcm, io_utils, mappings, metrics,
)

DATA_SEED = 42          # explicit seed replacing the original implicit state

METHODS = ["Traditional FCM", "LD-AMFCM (Exp)"]


def run(quick=False):
    experiment = "exp09_noise_ratio"

    ratios = config.NOISE_RATIO_GRID[:2] if quick else config.NOISE_RATIO_GRID

    io_utils.write_environment_report(experiment, extra={
        "data_seed": DATA_SEED,
        "init_seed": config.SEED_SINGLE_RUN,
        "ratios": ratios,
        "denominator": "clean sample count, n_noise = floor(ratio * n_clean)",
        "note": "explicit data seed replaces an implicit global state "
                "in the original notebook",
    })
    plt = io_utils.configure_matplotlib()

    sources = [
        ("Iris", load_iris().data, load_iris().target),
        ("Breast", load_breast_cancer().data, load_breast_cancer().target),
    ]

    np.random.seed(DATA_SEED)

    records = []

    for name, X_raw, y_raw in sources:
        for ratio in ratios:
            X_noisy, y_noisy = datasets.add_uniform_noise(
                X_raw, y_raw, ratio, rng=np.random
            )
            X = StandardScaler().fit_transform(X_noisy)
            n_clusters = len(np.unique(y_raw))
            n = X.shape[0]

            rho = density.local_density(X, k=config.K_NEIGHBORS,
                                        estimator=config.MAIN_DENSITY_ESTIMATOR)
            rho_bar = density.normalize_density(rho)
            w = density.density_weights(rho)

            for method in METHODS:
                if method == "Traditional FCM":
                    m_i, w_i = np.ones(n) * 2.0, np.ones(n)
                else:
                    m_i = mappings.fuzzifier_values("Exp", rho_bar)
                    w_i = w

                U = fcm.fcm_weighted(X, n_clusters, m_i, w_i,
                                     random_state=config.SEED_SINGLE_RUN,
                                     n_init=config.KMEANS_N_INIT)
                labels = U.argmax(axis=1)

                records.append({
                    "Dataset": name,
                    "Method": method,
                    "Noise_Ratio": "{:.0f}%".format(ratio * 100),
                    "Noise_Ratio_Value": ratio,
                    "N_Noise": int(round(ratio * len(y_raw))),
                    "N_Clean": int(len(y_raw)),
                    "N_Augmented": int(n),
                    "ARI": metrics.clean_ari(y_noisy, labels),
                    "DBI": metrics.clean_dbi(X, y_noisy, labels),
                    "Noise_Membership": metrics.noise_membership(U, y_noisy),
                    "Noise_AUC": metrics.noise_auc(U, y_noisy),
                })

        print("  {} done".format(name))

    df = pd.DataFrame(records)
    io_utils.save_table(df, experiment, "noise_ratio_sensitivity.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
    for ax, name in zip(axes, ["Iris", "Breast"]):
        part = df[df["Dataset"] == name]
        for method, marker in zip(METHODS, ["s", "o"]):
            sub = part[part["Method"] == method].sort_values("Noise_Ratio_Value")
            ax.plot(sub["Noise_Ratio_Value"] * 100, sub["ARI"],
                    marker=marker, linestyle="-", linewidth=2, markersize=7,
                    label=method)
        ax.set_xlabel("Injected points (% of clean samples)")
        ax.set_ylabel("ARI")
        ax.set_title(name)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
    fig.tight_layout()
    io_utils.save_figure(fig, experiment, "noise_ratio_sensitivity.png")
    plt.close(fig)

    print("\nExperiment 9 finished.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
