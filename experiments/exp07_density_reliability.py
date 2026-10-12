"""
Experiment 7 -- density-reliability stress test.

Contamination is not always peripheral.  Four synthetic scenarios separate the
cases in which a density-based reliability signal should work from those in
which it should not:

    Heterogeneous Density      equal-size legitimate clusters with very
                               different within-cluster spread (0.45/0.9/1.5)
    Sparse Legitimate Cluster  one cluster is genuinely sparse because it has
                               far fewer samples (220/220/40)
    Unequal Cluster Size       strongly imbalanced sizes (300/120/40) at
                               comparable density
    Mixed-Density Noise        30 peripheral points plus 30 compact points in
                               a locally dense region, so density alone cannot
                               identify all the contamination

Ten repetitions per scenario with the seed index carried by both streams
(data seed 42 + s and initialisation seed 42 + s).  This coupling is disclosed
in the seed register; it is a property of the original experiment.

Two diagnostics are properties of the density estimator alone and are
independent of any clustering method:

    Density AUC               ROC-AUC of -rho against the injected-point flag
    Noise/Clean Density       median density of injected points over that of
                              the clean samples

Per-cluster recall is computed after Hungarian matching of the predicted
clusters to the true ones.

Usage
-----
    python experiments/exp07_density_reliability.py
    python experiments/exp07_density_reliability.py --quick
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

METHODS = ["FCM", "DW-FCM", "FCM + m_i",
           "LD-AMFCM (Exp)", "LD-AMFCM (Rec)", "LD-AMFCM (Sig)"]

NUMERIC_COLUMNS = [
    "ARI", "DBI", "Noise_Membership", "Noise_AUC", "Noise_Density_AUC",
    "Noise_Clean_Density_Ratio", "Min_Cluster_Recall", "Macro_Cluster_Recall",
    "Median_Clean_Rho", "Median_Noise_Rho",
]

# Diagnostics that depend on the density estimator only, not on the method.
DENSITY_DIAGNOSTIC_COLUMNS = [
    "Noise_Density_AUC", "Noise_Clean_Density_Ratio",
    "Median_Clean_Rho", "Median_Noise_Rho",
]


def method_vectors(method, X, n_clusters, rho_bar, w):
    """The six variants of this experiment.

    The adaptive-only variant uses the exponential mapping here, unlike the
    ablation and the noise-count stress test which use the sigmoid mapping.
    """
    n = X.shape[0]

    if method == "FCM":
        return np.ones(n) * 2.0, np.ones(n)
    if method == "DW-FCM":
        return np.ones(n) * 2.0, w
    if method == "FCM + m_i":
        return mappings.fuzzifier_values("Exp", rho_bar), np.ones(n)
    if method == "LD-AMFCM (Exp)":
        return mappings.fuzzifier_values("Exp", rho_bar), w
    if method == "LD-AMFCM (Rec)":
        return mappings.fuzzifier_values("Rec", rho_bar), w
    if method == "LD-AMFCM (Sig)":
        return mappings.fuzzifier_values("Sig", rho_bar), w
    raise ValueError("Unknown method {!r}".format(method))


def run(quick=False):
    experiment = "exp07_density_reliability"
    seed_spec = config.SEEDS_DENSITY_STRESS

    seeds = seed_spec.data_seeds()
    if quick:
        seeds = seeds[:2]

    io_utils.write_environment_report(experiment, extra={
        "protocol": "regenerated input",
        "data_seed": "42 + s",
        "init_seed": "42 + s (coupled by construction, disclosed)",
        "n_runs": len(seeds),
        "scenarios": list(datasets.STRESS_SCENARIOS),
        "adaptive_mapping": "Exp",
    })
    plt = io_utils.configure_matplotlib()

    records = []

    for scenario_name, generator in datasets.STRESS_SCENARIOS.items():
        print("\nscenario: {}".format(scenario_name))

        for seed in seeds:
            X_raw, y_true = generator(seed)
            X = datasets.standardize(X_raw)
            n_clusters = datasets.n_clusters_from_labels(y_true)
            mask_clean = y_true >= 0

            rho = density.local_density(
                X, k=config.K_NEIGHBORS, estimator=config.MAIN_DENSITY_ESTIMATOR
            )
            rho_bar = density.normalize_density(rho)
            w = density.density_weights(rho)

            density_diagnostics = {
                "Noise_Density_AUC": metrics.density_auc(rho, y_true),
                "Noise_Clean_Density_Ratio": metrics.density_ratio(rho, y_true),
                "Median_Clean_Rho": float(np.median(rho[mask_clean])),
                "Median_Noise_Rho": float(np.median(rho[~mask_clean])),
            }

            for method in METHODS:
                m_i, w_i = method_vectors(method, X, n_clusters, rho_bar, w)
                U = fcm.fcm_weighted(
                    X, n_clusters, m_i, w_i,
                    random_state=seed, n_init=config.KMEANS_N_INIT,
                )
                labels = U.argmax(axis=1)

                _recalls, min_recall, macro_recall = metrics.cluster_recalls(
                    y_true, labels, n_clusters
                )

                row = {
                    "Scenario": scenario_name,
                    "Seed": seed,
                    "Method": method,
                    "ARI": metrics.clean_ari(y_true, labels),
                    # This cell reported the raw Davies-Bouldin value, whose
                    # degenerate case is 0 rather than NaN.
                    "DBI": metrics.davies_bouldin_index(
                        X[mask_clean], labels[mask_clean]
                    ),
                    "Noise_Membership": metrics.noise_membership(U, y_true),
                    "Noise_AUC": metrics.noise_auc(U, y_true),
                    "Min_Cluster_Recall": min_recall,
                    "Macro_Cluster_Recall": macro_recall,
                }
                row.update(density_diagnostics)
                records.append(row)

        print("  {} repetitions done".format(len(seeds)))

    raw = pd.DataFrame(records)[
        ["Scenario", "Seed", "Method"] + NUMERIC_COLUMNS
    ]
    io_utils.save_table(raw, experiment, "density_reliability_stress_raw.csv")

    summary = pd.DataFrame([
        {
            "Scenario": scenario,
            "Method": method,
            **{
                "{}_Mean".format(column): float(group[column].mean())
                for column in NUMERIC_COLUMNS
            },
            **{
                "{}_Std".format(column): float(group[column].std())
                for column in NUMERIC_COLUMNS
            },
        }
        for (scenario, method), group in raw.groupby(["Scenario", "Method"])
    ])
    io_utils.save_table(summary, experiment,
                        "density_reliability_stress_summary.csv")

    # Publication-style table with mean +/- std strings.
    publication = []
    for (scenario, method), group in raw.groupby(["Scenario", "Method"]):
        publication.append({
            "Scenario": scenario,
            "Method": method,
            "ARI": io_utils.mean_std_str(group["ARI"].mean(),
                                          group["ARI"].std()),
            "Noise Membership": io_utils.mean_std_str(
                group["Noise_Membership"].mean(),
                group["Noise_Membership"].std()),
            "Noise AUC": io_utils.mean_std_str(group["Noise_AUC"].mean(),
                                                group["Noise_AUC"].std()),
            "Density AUC": io_utils.mean_std_str(
                group["Noise_Density_AUC"].mean(),
                group["Noise_Density_AUC"].std()),
            "Noise/Clean Density": io_utils.mean_std_str(
                group["Noise_Clean_Density_Ratio"].mean(),
                group["Noise_Clean_Density_Ratio"].std()),
            "Min Cluster Recall": io_utils.mean_std_str(
                group["Min_Cluster_Recall"].mean(),
                group["Min_Cluster_Recall"].std()),
        })
    io_utils.save_table(pd.DataFrame(publication), experiment,
                        "density_reliability_stress_publication_table.csv")

    # Density diagnostics are method-independent; report them once per scenario.
    diagnostics = raw[
        ["Scenario", "Seed"] + DENSITY_DIAGNOSTIC_COLUMNS
    ].drop_duplicates()
    diagnostics_summary = diagnostics.groupby("Scenario").agg(
        {column: ["mean", "std"] for column in DENSITY_DIAGNOSTIC_COLUMNS}
    )
    io_utils.save_table(diagnostics_summary, experiment,
                        "density_diagnostics_summary.csv", index=True)

    # ---- figures -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.boxplot(
        [raw[raw["Scenario"] == scenario]["ARI"].values
         for scenario in datasets.STRESS_SCENARIOS],
        labels=list(datasets.STRESS_SCENARIOS),
    )
    ax.set_xticklabels(list(datasets.STRESS_SCENARIOS), rotation=15, ha="right")
    ax.set_ylabel("ARI")
    ax.set_title("Clustering performance under density-reliability stress")
    fig.tight_layout()
    io_utils.save_figure(fig, experiment, "stress_ari_boxplot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    ax.boxplot(
        [diagnostics[diagnostics["Scenario"] == scenario]["Noise_Density_AUC"].values
         for scenario in datasets.STRESS_SCENARIOS],
        labels=list(datasets.STRESS_SCENARIOS),
    )
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticklabels(list(datasets.STRESS_SCENARIOS), rotation=15, ha="right")
    ax.set_ylabel("Noise density AUC")
    ax.set_title("Can local density identify the contamination?")
    fig.tight_layout()
    io_utils.save_figure(fig, experiment, "stress_density_auc_boxplot.png")
    plt.close(fig)

    print("\nExperiment 7 finished.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
