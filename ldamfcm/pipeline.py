"""
Shared pipeline helpers.

Every method that operates in the input space is the same solver driven by a
per-sample fuzzifier vector ``m_i`` and a sample-weight vector ``w_i``.  This
module centralises the construction of those vectors so that the experiment
scripts cannot drift apart from one another.
"""

import numpy as np

from . import config, datasets, density, fcm, mappings, metrics, models


def prepare(X_raw, k=config.K_NEIGHBORS,
            estimator=config.MAIN_DENSITY_ESTIMATOR,
            algorithm="auto", n_jobs=1):
    """Standardise a raw benchmark and derive its density quantities.

    Returns
    -------
    X : ndarray
        Standardised augmented matrix.
    rho : ndarray
        Raw local density.
    rho_bar : ndarray
        Min-max normalised density.
    w : ndarray
        Normalised sample weights.
    """
    X = datasets.standardize(X_raw)
    rho = density.local_density(
        X, k=k, estimator=estimator, algorithm=algorithm, n_jobs=n_jobs
    )
    return X, rho, density.normalize_density(rho), density.density_weights(rho)


def method_vectors(method, X, n_clusters, rho_bar, w, adaptive_mapping="Sig"):
    """Return ``(m_i, w_i)`` for a labelled input-space method.

    ``adaptive_mapping`` selects the mapping used by the adaptive-only variant
    ``FCM + m_i``.  It is the sigmoid mapping in the ablation and in the
    noise-count stress test, and the exponential mapping in the
    density-reliability stress test.
    """
    n_samples = X.shape[0]

    if method == config.METHOD_FCM:
        return np.ones(n_samples) * 2.0, np.ones(n_samples)

    if method == config.METHOD_DW_FCM:
        return np.ones(n_samples) * 2.0, w

    if method == config.METHOD_FCM_MI:
        return mappings.fuzzifier_values(adaptive_mapping, rho_bar), np.ones(n_samples)

    if method == config.METHOD_LD_EXP:
        return mappings.fuzzifier_values("Exp", rho_bar), w

    if method == config.METHOD_LD_REC:
        return mappings.fuzzifier_values("Rec", rho_bar), w

    if method == config.METHOD_LD_SIG:
        return mappings.fuzzifier_values("Sig", rho_bar), w

    if method == config.METHOD_MC_FCM:
        return models.mc_fcm_fuzzifier(X, n_clusters), np.ones(n_samples)

    raise ValueError("Unknown input-space method {!r}".format(method))


def solve(method, X, n_clusters, rho_bar, w, init_seed,
          n_init=config.KMEANS_N_INIT, adaptive_mapping="Sig",
          return_diagnostics=False, convergence=fcm.PROTOTYPE_FROBENIUS):
    """Build the vectors for ``method`` and run the weighted solver."""
    m_i, w_i = method_vectors(
        method, X, n_clusters, rho_bar, w, adaptive_mapping=adaptive_mapping
    )
    return fcm.fcm_weighted(
        X, n_clusters, m_i, w_i,
        random_state=init_seed,
        n_init=n_init,
        convergence=convergence,
        return_diagnostics=return_diagnostics,
    )


def evaluate(U, X, y_true, rho=None, with_recall=False):
    """Standard evaluation block shared by the comparison experiments."""
    n_clusters = datasets.n_clusters_from_labels(y_true)
    labels = U.argmax(axis=1)

    result = {
        "ARI": metrics.clean_ari(y_true, labels),
        "DBI": metrics.clean_dbi(X, y_true, labels),
        "Noise_Membership": metrics.noise_membership(U, y_true),
        "Noise_AUC": metrics.noise_auc(U, y_true),
    }

    if rho is not None:
        result["Noise_Density_AUC"] = metrics.density_auc(rho, y_true)
        result["Noise_Clean_Density_Ratio"] = metrics.density_ratio(rho, y_true)

    if with_recall:
        _recalls, min_recall, macro_recall = metrics.cluster_recalls(
            y_true, labels, n_clusters
        )
        result["Min_Cluster_Recall"] = min_recall
        result["Macro_Cluster_Recall"] = macro_recall

    return result
