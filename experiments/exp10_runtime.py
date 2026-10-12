"""
Experiment 10 -- preprocessing and iteration cost.

The point of this table is that the density-derived quantities are a one-off
preprocessing cost, not a per-iteration cost: once rho, rho_bar and the
weights exist, the iterative part of LD-AMFCM is the same cost as plain FCM.

Reported on the contaminated Breast Cancer benchmark:

    Prep_Time(s)   density estimation plus the mapping evaluation
    Iter_Time(s)   the alternating optimisation
    Total(s)       prep + iteration

Only the relative magnitudes are meaningful; absolute values are machine
specific.  Wall-clock seconds via ``time.time()``.

Reproducibility note
--------------------
As in Exp. 9, the original notebook drew its contamination without resetting
the generator.  This script resets it explicitly.  See
docs/KNOWN_VARIATIONS.md.

Usage
-----
    python experiments/exp10_runtime.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ldamfcm import (  # noqa: E402
    config, density, fcm, io_utils, mappings, metrics,
)

DATA_SEED = 42
N_OUTLIERS = 30


def run():
    experiment = "exp10_runtime"

    io_utils.write_environment_report(experiment, extra={
        "data_seed": DATA_SEED,
        "init_seed": config.SEED_SINGLE_RUN,
        "timer": "time.time(), wall-clock seconds",
        "note": "relative timings only; absolute values are machine specific",
    })

    breast = load_breast_cancer()
    X_clean, y_clean = breast.data, breast.target

    np.random.seed(DATA_SEED)
    outliers = np.random.uniform(
        X_clean.min(axis=0), X_clean.max(axis=0),
        (N_OUTLIERS, X_clean.shape[1]),
    )
    X = StandardScaler().fit_transform(np.vstack([X_clean, outliers]))
    n_clusters = len(np.unique(y_clean))
    n = X.shape[0]

    results = []

    # ---- plain FCM: no preprocessing at all ----------------------------
    started = time.time()
    fcm.fcm_weighted(X, n_clusters, np.ones(n) * 2.0, np.ones(n),
                     random_state=config.SEED_SINGLE_RUN,
                     n_init=config.KMEANS_N_INIT)
    iteration_time = time.time() - started
    results.append({
        "Method": config.METHOD_FCM,
        "Prep_Time(s)": 0.0,
        "Iter_Time(s)": iteration_time,
        "Total(s)": iteration_time,
    })

    # ---- density estimation, timed once --------------------------------
    started = time.time()
    rho = density.local_density(X, k=config.K_NEIGHBORS,
                                estimator=config.MAIN_DENSITY_ESTIMATOR)
    base_prep_time = time.time() - started

    rho_bar = density.normalize_density(rho)
    w = density.density_weights(rho)

    variants = [
        (config.METHOD_DW_FCM, np.ones(n) * 2.0, w, "DW-FCM"),
        (config.METHOD_LD_EXP, mappings.fuzzifier_values("Exp", rho_bar), w, "Exp"),
        (config.METHOD_LD_REC, mappings.fuzzifier_values("Rec", rho_bar), w, "Rec"),
        (config.METHOD_LD_SIG, mappings.fuzzifier_values("Sig", rho_bar), w, "Sig"),
    ]

    for name, m_i, w_i, mapping_name in variants:
        if name == config.METHOD_DW_FCM:
            prep_time = base_prep_time
        else:
            started = time.time()
            mappings.fuzzifier_values(mapping_name, rho_bar)
            prep_time = base_prep_time + (time.time() - started)

        started = time.time()
        fcm.fcm_weighted(X, n_clusters, m_i, w_i,
                         random_state=config.SEED_SINGLE_RUN,
                         n_init=config.KMEANS_N_INIT)
        iteration_time = time.time() - started

        results.append({
            "Method": name,
            "Prep_Time(s)": prep_time,
            "Iter_Time(s)": iteration_time,
            "Total(s)": prep_time + iteration_time,
        })

    df = pd.DataFrame(results)
    io_utils.save_table(df, experiment, "runtime_comparison.csv")

    print("\nruntime comparison (Breast Cancer + outliers, n={})".format(n))
    print(df.to_string(index=False))

    print("\nExperiment 10 finished.")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="accepted for interface uniformity; this "
                             "experiment has no grid to reduce")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
