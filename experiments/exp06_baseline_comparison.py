"""
Experiment 6 -- comparison against published baselines.

This is the main result table.  Five augmented benchmarks, thirteen methods,
ten repetitions.

Protocol
--------
Frozen input: the contaminated matrix of each benchmark is generated once
under the data seed and then held fixed.  The ten repetitions vary the
initialisation seed only (0..9), and random_state is passed explicitly to
every method that takes one, so no method secretly draws from a global stream.

Nine of the methods operate in the input space and share one solver.  For
those, the initialisation seed is common, so within a matched run FCM, DW-FCM,
the three LD-AMFCM mappings and MC-FCM all start from identical initial
prototypes.  AWDFCM, Adaptive FCM-GE and EM-IFCM follow their own published
initialisation rules and therefore start from different points by design.

Metrics, all computed on the uncontaminated samples except where noted:

    ARI                agreement with the known partition
    DBI                Davies-Bouldin index of the produced partition
    Noise_Membership   mean largest membership over the injected points
                       (lower is better)
    Noise_AUC          ROC-AUC of membership uncertainty against the
                       injected-point flag over all samples

DBSCAN is reported as a reference-tuned upper bound: its two hyperparameters
are searched over a declared grid and the configuration with the best ARI on
the clean ground truth is reported.  It is not an untuned competitor and is
labelled as such.

``--runs 1`` reproduces the single-run version of this table with
initialisation seed 42.

Usage
-----
    python experiments/exp06_baseline_comparison.py
    python experiments/exp06_baseline_comparison.py --runs 1
    python experiments/exp06_baseline_comparison.py --quick
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldamfcm import (  # noqa: E402
    config, datasets, density, io_utils, metrics, models, pipeline,
)

# Methods that are the shared weighted solver with different (m_i, w_i).
INPUT_SPACE_METHODS = [
    config.METHOD_FCM,
    config.METHOD_DW_FCM,
    config.METHOD_LD_EXP,
    config.METHOD_LD_REC,
    config.METHOD_LD_SIG,
    config.METHOD_MC_FCM,
]

# Methods that follow their own published optimisation.
PUBLICATION_METHODS = [
    config.METHOD_AWDFCM,
    config.METHOD_AFCM_GE,
    config.METHOD_EM_IFCM,
]

# Methods that are not fuzzy c-means at all.
NON_STANDARD_METHODS = [
    config.METHOD_KMEANS,
    config.METHOD_GMM,
    config.METHOD_DBSCAN,
    config.METHOD_AGGLOMERATIVE,
]

ALL_METHODS = INPUT_SPACE_METHODS + PUBLICATION_METHODS + NON_STANDARD_METHODS


def run_publication_method(method, X, n_clusters, init_seed):
    """Dispatch the three methods that implement their own published update."""
    if method == config.METHOD_AWDFCM:
        gamma, eta = models.awdfcm_dimensional_settings(X.shape[1])
        return models.awdfcm(
            X, n_clusters,
            theta=config.AWDFCM_THETA,
            gamma=gamma,
            eta=eta,
            alpha=config.AWDFCM_ALPHA,
            beta=config.AWDFCM_BETA,
            epsilon=None,
            max_iter=config.AWDFCM_MAX_ITER,
            tol=config.AWDFCM_TOL,
            random_state=init_seed,
        )

    if method == config.METHOD_AFCM_GE:
        return models.adaptive_fcm_ge(
            X, n_clusters,
            lam=config.AFCM_GE_LAMBDA,
            k=config.AFCM_GE_K,
            sigma=config.AFCM_GE_SIGMA,
            max_iter=config.AFCM_GE_MAX_ITER,
            tol=config.AFCM_GE_TOL,
            random_state=init_seed,
        )

    if method == config.METHOD_EM_IFCM:
        return models.em_ifcm(
            X, n_clusters,
            m=config.EM_IFCM_M,
            delta_prime=config.EM_IFCM_DELTA_PRIME,
            max_iter=config.EM_IFCM_MAX_ITER,
            tol=config.EM_IFCM_TOL,
            random_state=init_seed,
        )

    raise ValueError("Unknown publication method {!r}".format(method))


def run_dbscan(X, y_true, n_clusters):
    """Search the declared DBSCAN grid and keep the best clean ARI."""
    mask = y_true >= 0
    best_model = None
    best_ari = -1.0

    for eps in config.DBSCAN_EPS_GRID:
        for min_samples in config.DBSCAN_MIN_SAMPLES_GRID:
            model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            if len(np.unique(model.labels_[mask])) > 1:
                ari = metrics.clean_ari(y_true, model.labels_)
                if ari > best_ari:
                    best_ari = ari
                    best_model = model

    if best_model is None:
        return None

    is_noise = (y_true < 0).astype(int)
    return {
        "ARI": float(best_ari),
        "DBI": metrics.clean_dbi(X, y_true, best_model.labels_),
        "Noise_Membership": 0.0,
        "Noise_AUC": float(roc_auc_score(
            is_noise, (best_model.labels_ == -1).astype(int)
        )),
    }


def run_non_standard(method, X, y_true, n_clusters, init_seed):
    """Evaluate the four methods that are not plain weighted FCM."""
    if method == config.METHOD_KMEANS:
        labels = KMeans(n_clusters=n_clusters, init="k-means++",
                        n_init=config.KMEANS_N_INIT,
                        random_state=init_seed).fit(X).labels_
        return {
            "ARI": metrics.clean_ari(y_true, labels),
            "DBI": metrics.clean_dbi(X, y_true, labels),
            "Noise_Membership": 1.0,
            "Noise_AUC": 0.5,
        }

    if method == config.METHOD_GMM:
        gmm = GaussianMixture(n_components=n_clusters,
                              random_state=init_seed).fit(X)
        labels = gmm.predict(X)
        probabilities = gmm.predict_proba(X)
        is_noise = (y_true < 0).astype(int)

        return {
            "ARI": metrics.clean_ari(y_true, labels),
            "DBI": metrics.clean_dbi(X, y_true, labels),
            "Noise_Membership": float(
                probabilities[y_true < 0].max(axis=1).mean()
            ) if np.any(y_true < 0) else 0.0,
            "Noise_AUC": float(roc_auc_score(
                is_noise, 1.0 - probabilities.max(axis=1)
            )),
        }

    if method == config.METHOD_AGGLOMERATIVE:
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit(X).labels_
        return {
            "ARI": metrics.clean_ari(y_true, labels),
            "DBI": metrics.clean_dbi(X, y_true, labels),
            "Noise_Membership": 1.0,
            "Noise_AUC": 0.5,
        }

    raise ValueError("Unknown non-standard method {!r}".format(method))


def run(quick=False, runs=None):
    experiment = "exp06_baseline_comparison"
    seed_spec = config.SEEDS_MAIN_COMPARISON

    n_runs = runs if runs is not None else seed_spec.n_runs
    if quick:
        n_runs = min(n_runs, 2)

    init_seeds = (
        [config.SEED_SINGLE_RUN] if n_runs == 1
        else list(range(n_runs))
    )

    io_utils.write_environment_report(experiment, extra={
        "protocol": "frozen input",
        "data_seed": seed_spec.data_seed,
        "init_seeds": init_seeds,
        "n_runs": len(init_seeds),
        "standard_methods_n_init": config.KMEANS_N_INIT,
        "dbscan_grid": {
            "eps": config.DBSCAN_EPS_GRID,
            "min_samples": config.DBSCAN_MIN_SAMPLES_GRID,
            "selection": "best ARI on clean ground truth (reference-tuned)",
        },
    })
    plt = io_utils.configure_matplotlib()

    benchmarks = datasets.get_benchmarks(
        data_seed=seed_spec.data_seed, local_rng=False
    )
    if quick:
        benchmarks = benchmarks[:1]

    per_run_records = []

    for benchmark_name, X_raw, y_true in benchmarks:
        X = datasets.standardize(X_raw)
        n_clusters = datasets.n_clusters_from_labels(y_true)
        is_noise = (y_true < 0).astype(int)

        print("\nbenchmark: {} (n={}, p={}, C={})".format(
            benchmark_name, X.shape[0], X.shape[1], n_clusters))

        rho = density.local_density(X, k=config.K_NEIGHBORS,
                                    estimator=config.MAIN_DENSITY_ESTIMATOR)
        rho_bar = density.normalize_density(rho)
        w = density.density_weights(rho)

        for init_seed in init_seeds:
            for method in INPUT_SPACE_METHODS:
                U = pipeline.solve(
                    method, X, n_clusters, rho_bar, w,
                    init_seed=init_seed, n_init=config.KMEANS_N_INIT,
                )
                result = pipeline.evaluate(U, X, y_true)
                result.update({
                    "Dataset": benchmark_name,
                    "Method": method,
                    "Run": init_seed,
                })
                per_run_records.append(result)

            for method in PUBLICATION_METHODS:
                U = run_publication_method(method, X, n_clusters, init_seed)
                result = pipeline.evaluate(U, X, y_true)
                result.update({
                    "Dataset": benchmark_name,
                    "Method": method,
                    "Run": init_seed,
                })
                per_run_records.append(result)

            for method in NON_STANDARD_METHODS:
                if method == config.METHOD_DBSCAN:
                    result = run_dbscan(X, y_true, n_clusters)
                    if result is None:
                        continue
                else:
                    result = run_non_standard(method, X, y_true, n_clusters,
                                              init_seed)
                result.update({
                    "Dataset": benchmark_name,
                    "Method": method,
                    "Run": init_seed,
                })
                per_run_records.append(result)

        print("  {} repetitions x {} methods done".format(
            len(init_seeds), len(ALL_METHODS)))

    per_run = pd.DataFrame(per_run_records)
    io_utils.save_table(per_run, experiment,
                        "strong_fcm_baselines_per_run.csv")

    # ---- aggregate over the repetitions --------------------------------
    aggregated = []

    for (dataset, method), group in per_run.groupby(["Dataset", "Method"]):
        entry = {
            "Dataset": dataset,
            "Method": method,
            "N_Runs": len(group),
        }

        for metric in ("ARI", "DBI", "Noise_Membership", "Noise_AUC"):
            values = group[metric].values.astype(float)
            if np.all(np.isnan(values)):
                mean, std = np.nan, np.nan
            else:
                mean = float(np.nanmean(values))
                std = (float(np.nanstd(values, ddof=1))
                       if len(values) > 1 else 0.0)
            entry["{}_Mean".format(metric)] = mean
            entry["{}_Std".format(metric)] = std

        aggregated.append(entry)

    df = pd.DataFrame(aggregated)

    # DBSCAN is deterministic under the declared protocol, so its reported
    # spread is zero by construction.
    deterministic = [config.METHOD_DBSCAN, config.METHOD_AGGLOMERATIVE]
    for column in ("ARI", "DBI", "Noise_Membership", "Noise_AUC"):
        mask = df["Method"].isin(deterministic)
        df.loc[mask, "{}_Std".format(column)] = 0.0

    columns = ["Dataset", "Method",
               "ARI_Mean", "ARI_Std",
               "DBI_Mean", "DBI_Std",
               "Noise_Membership_Mean", "Noise_Membership_Std",
               "Noise_AUC_Mean", "Noise_AUC_Std",
               "N_Runs"]
    df = df[columns]
    io_utils.save_table(df, experiment, "strong_fcm_baselines_results.csv")

    overall = df.groupby("Method").agg({
        "ARI_Mean": ["mean", "std"],
        "DBI_Mean": ["mean", "std"],
        "Noise_Membership_Mean": ["mean", "std"],
        "Noise_AUC_Mean": ["mean", "std"],
    })
    io_utils.save_table(overall, experiment,
                        "strong_fcm_baselines_overall_summary.csv",
                        index=True)

    # ---- directly usable mean +/- std tables ---------------------------
    display = df.copy()
    for metric in ("ARI", "DBI", "Noise_Membership", "Noise_AUC"):
        display[metric] = [
            io_utils.mean_std_str(m, s) if not np.isnan(m) else "nan"
            for m, s in zip(display["{}_Mean".format(metric)],
                            display["{}_Std".format(metric)])
        ]
    display = display[["Dataset", "Method", "ARI", "DBI",
                       "Noise_Membership", "Noise_AUC", "N_Runs"]]
    io_utils.save_table(display, experiment,
                        "strong_fcm_baselines_results_mean_std.csv")

    overall_display = (
        df.groupby("Method")
        .agg({"ARI_Mean": "mean", "ARI_Std": "mean",
              "DBI_Mean": "mean", "DBI_Std": "mean",
              "Noise_Membership_Mean": "mean", "Noise_Membership_Std": "mean",
              "Noise_AUC_Mean": "mean", "Noise_AUC_Std": "mean"})
        .reset_index()
    )
    for metric in ("ARI", "DBI", "Noise_Membership", "Noise_AUC"):
        overall_display[metric] = [
            io_utils.mean_std_str(m, s)
            for m, s in zip(overall_display["{}_Mean".format(metric)],
                            overall_display["{}_Std".format(metric)])
        ]
    overall_display = overall_display[
        ["Method", "ARI", "DBI", "Noise_Membership", "Noise_AUC"]
    ]
    io_utils.save_table(overall_display, experiment,
                        "strong_fcm_baselines_overall_summary_mean_std.csv")

    # ---- headline figure ------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    datasets_order = df["Dataset"].unique()
    methods_order = (
        df.groupby("Method")["ARI_Mean"].mean().sort_values(ascending=False).index
    )
    width = 0.8 / len(methods_order)
    positions = np.arange(len(datasets_order))

    for i, method in enumerate(methods_order):
        values = [
            float(df[(df["Dataset"] == ds) & (df["Method"] == method)]
                  ["ARI_Mean"].mean())
            for ds in datasets_order
        ]
        ax.bar(positions + i * width, values, width, label=method)

    ax.set_xticks(positions + 0.4 - width / 2)
    ax.set_xticklabels(datasets_order, rotation=15, ha="right")
    ax.set_ylabel("ARI (mean over repetitions)")
    ax.set_title("Method comparison across benchmarks")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    io_utils.save_figure(fig, experiment, "method_comparison_ari.png")
    plt.close(fig)

    print("\nExperiment 6 finished ({} runs per benchmark).".format(len(init_seeds)))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--runs", type=int, default=None,
                        help="override the number of repetitions; 1 reproduces "
                             "the single-run table with seed 42")
    args = parser.parse_args()
    run(quick=args.quick, runs=args.runs)


if __name__ == "__main__":
    main()
