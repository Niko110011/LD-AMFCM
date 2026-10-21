"""
Experiment 2 -- noise-count stress test.

Contamination is injected at seven absolute levels into the Breast Cancer
benchmark.  For every level the contaminated matrix is regenerated for each
repetition, and the two seed streams are deliberately offset:

    data seed  = 100 + s
    init seed  = s          s = 0, ..., 9

so the data seed and the initialisation seed are never confounded.

Three variants are compared, isolating the two ingredients of the method:

    FCM          constant fuzzifier, uniform weights
    FCM + m_i    sample-wise fuzzifier, uniform weights
    LD-AMFCM     sample-wise fuzzifier, density-derived weights

The fuzzifier is the sigmoid mapping with gamma = 12.

Because the contamination is regenerated, this experiment is a
*regenerated-input* protocol and is reported separately from the frozen-input
comparison.  Its contamination ratio has an additive structure: the number of
injected points is held constant while the clean sample count does not change,
so the ratio over the augmented matrix runs from 6.6 % (40 points) to 22.0 %
(160 points).

The clean ground-truth labels are used for the ARI, and the injected points
are excluded from it.

Usage
-----
    python experiments/exp02_noise_count_stress.py
    python experiments/exp02_noise_count_stress.py --quick
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldamfcm import (  # noqa: E402
    config, datasets, density, fcm, io_utils, mappings, metrics,
)

METHODS = ["FCM", "FCM + m_i", "LD-AMFCM"]

MAPPING = "Sig"


def build(noise_count, data_seed):
    """Breast Cancer plus ``noise_count`` uniform points, standardised."""
    breast = load_breast_cancer()
    X_clean, y_clean = breast.data, breast.target

    np.random.seed(data_seed)
    noise = np.random.uniform(
        X_clean.min(axis=0), X_clean.max(axis=0),
        (noise_count, X_clean.shape[1]),
    )

    X_mixed = np.vstack([X_clean, noise])
    y_mixed = np.hstack([y_clean, config.NOISE_LABEL * np.ones(noise_count)])

    return StandardScaler().fit_transform(X_mixed), y_mixed


def run(quick=False, legacy_membership=False):
    experiment = "exp02_noise_count_stress"
    seed_spec = config.SEEDS_NOISE_COUNT

    membership_form = (fcm.MEMBERSHIP_NEGATIVE_POWER if legacy_membership
                       else fcm.MEMBERSHIP_RATIO)

    io_utils.write_environment_report(experiment, extra={
        "protocol": "regenerated input",
        "data_seed": "100 + s",
        "init_seed": "s",
        "n_runs": seed_spec.n_runs,
        "noise_counts": config.NOISE_COUNT_GRID,
        "adaptive_mapping": MAPPING,
        "membership_form": membership_form,
    })
    plt = io_utils.configure_matplotlib()

    noise_levels = config.NOISE_COUNT_GRID[:3] if quick else config.NOISE_COUNT_GRID
    init_seeds = seed_spec.init_seeds()
    if quick:
        init_seeds = init_seeds[:2]

    records = []

    print("protocol: data seed 100+s, init seed s, {} runs".format(len(init_seeds)))

    for noise_count in noise_levels:
        for data_seed, init_seed in zip(
            (seed_spec.data_seed + i for i in range(len(init_seeds))),
            init_seeds,
        ):
            X, y_true = build(noise_count, data_seed)
            n_clusters = datasets.n_clusters_from_labels(y_true)
            mask = y_true >= 0

            rho = density.local_density(X, k=config.K_NEIGHBORS,
                                        estimator=config.MAIN_DENSITY_ESTIMATOR)
            rho_bar = density.normalize_density(rho)
            w = density.density_weights(rho)

            fuzzifiers = {
                "FCM": (np.ones(len(X)) * 2.0, np.ones(len(X))),
                "FCM + m_i": (mappings.fuzzifier_values(MAPPING, rho_bar),
                              np.ones(len(X))),
                "LD-AMFCM": (mappings.fuzzifier_values(MAPPING, rho_bar), w),
            }

            for method in METHODS:
                m_i, w_i = fuzzifiers[method]
                U = fcm.fcm_weighted(X, n_clusters, m_i, w_i,
                                     random_state=init_seed,
                                     n_init=config.n_init_for(
                                         "exp02_noise_count_stress"),
                                     membership_form=membership_form)
                ari = metrics.clean_ari(y_true, U.argmax(axis=1))

                records.append({
                    "Noise_Count": noise_count,
                    "Data_Seed": data_seed,
                    "Init_Seed": init_seed,
                    "Method": method,
                    "ARI": ari,
                })

        print("  noise count {:>4d} done".format(noise_count))

    raw = pd.DataFrame(records)
    suffix = "_legacy" if legacy_membership else ""
    io_utils.save_table(raw, experiment,
                        "noise_count_stress_raw{}.csv".format(suffix))

    summary = (
        raw.groupby(["Noise_Count", "Method"])["ARI"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "ARI_Mean", "std": "ARI_Std"})
    )
    io_utils.save_table(
        summary, experiment,
        "noise_count_stress_summary{}.csv".format(suffix))

    # ---- figure --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    colors = ["#7f8c8d", "#2980b9", "#c0392b"]
    linestyles = ["--", "-.", "-"]
    markers = ["s", "^", "o"]

    for i, method in enumerate(METHODS):
        part = summary[summary["Method"] == method].sort_values("Noise_Count")
        ax.plot(part["Noise_Count"], part["ARI_Mean"], label=method,
                color=colors[i], linestyle=linestyles[i], marker=markers[i],
                linewidth=2, markersize=7)
        ax.fill_between(
            part["Noise_Count"],
            part["ARI_Mean"] - part["ARI_Std"].fillna(0),
            part["ARI_Mean"] + part["ARI_Std"].fillna(0),
            color=colors[i], alpha=0.15,
        )

    ax.set_xlabel("Noise count", fontsize=12)
    ax.set_ylabel("ARI mean", fontsize=12)
    ax.set_title("Effect of contamination level on ARI", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    fig.tight_layout()
    io_utils.save_figure(
        fig, experiment,
        "robustness_stress_test{}.png".format(suffix))
    plt.close(fig)

    print("\nExperiment 2 finished.")
    return summary


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
