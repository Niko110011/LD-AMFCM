"""
Weighted fuzzy c-means solvers.

The original notebook used one solver for every method that operates in the
input space; FCM, DW-FCM, FCM+m_i, MC-FCM and LD-AMFCM differ only in the
per-sample fuzzifier vector ``m_i`` and the sample-weight vector ``w_i``:

    ==================  ==========================  ====================
    method              m_i                         w_i
    ==================  ==========================  ====================
    FCM (Baseline)      constant 2.0                constant 1.0
    DW-FCM              constant 2.0                rho / sum(rho)
    FCM + m_i           mapping(rho_bar)            constant 1.0
    MC-FCM              MC-FCM fuzzifier            constant 1.0
    LD-AMFCM            mapping(rho_bar)            rho / sum(rho)
    ==================  ==========================  ====================

Two solvers appear in the notebook:

``fcm_weighted``
    Prototype initialisation (k-means++), squared distances, prototype-change
    convergence.  This is the headline solver and covers every experiment
    except the m_min / m_max sweeps.

``fcm_weighted_random_membership``
    Random membership initialisation with the prototypes initialised to zero,
    Euclidean (non-squared) distances, membership-change convergence.  Used by
    the m_min / m_max sweeps only.

Both return hard labels via ``U.argmax(axis=1)``.
"""

import time

import numpy as np
from sklearn.cluster import KMeans

from .config import ETA, MAX_ITER, TOL

PROTOTYPE_FROBENIUS = "prototype_frobenius"
PROTOTYPE_MAXSHIFT = "prototype_maxshift"

CONVERGENCE_CRITERIA = (PROTOTYPE_FROBENIUS, PROTOTYPE_MAXSHIFT)


def _membership_update_from_squared_distances(dist_sq, m_i, n_samples, n_clusters):
    """Standard fuzzy membership for a per-sample fuzzifier.

    ``u_ij = 1 / sum_k (d_ij / d_ik) ** (1 / (m_i - 1))``

    where ``d`` are the squared distances including the additive stabiliser.
    This is algebraically identical to the ratio of the negative powers used
    by the earlier cells of the notebook, and is numerically better behaved.

    The pairwise distance ratio is an outer division: for row ``i`` the entry
    ``[j, k]`` is ``d_ij / d_ik``, and the membership of ``j`` sums over ``k``.
    """
    # exponent[i] = 1 / (m_i - 1), applied to every ratio of sample i
    exponent = 1.0 / (m_i - 1.0)

    # shape (n_samples, n_clusters, n_clusters): ratios[i, j, k] = d_ij / d_ik
    ratios = dist_sq[:, :, None] / dist_sq[:, None, :]

    return 1.0 / np.sum(ratios ** exponent[:, None, None], axis=2)


def _prototype_update(X, U, m_i, w_i, n_clusters):
    """Weighted fuzzy prototype update, denominator guarded by 1e-12."""
    v_new = np.zeros((n_clusters, X.shape[1]), dtype=float)
    for j in range(n_clusters):
        weighted = w_i * (U[:, j] ** m_i)
        v_new[j] = (
            (weighted[:, None] * X).sum(axis=0) / (weighted.sum() + 1e-12)
        )
    return v_new


def fcm_weighted(
    X,
    n_clusters,
    m_i,
    w_i,
    random_state=None,
    n_init=1,
    max_iter=MAX_ITER,
    tol=TOL,
    eta=ETA,
    convergence=PROTOTYPE_FROBENIUS,
    return_diagnostics=False,
):
    """Weighted fuzzy c-means with k-means++ prototype initialisation.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    n_clusters : int
    m_i : ndarray, shape (n_samples,)
        Per-sample fuzzifier; must stay strictly above 1.
    w_i : ndarray, shape (n_samples,)
        Non-negative sample weights.
    random_state : int or None
        Seed of the k-means++ initialisation.  ``None`` draws from the global
        NumPy random state, which is what the regenerated-input experiment of
        the original notebook relied on.  Passing an explicit integer is the
        reproducible form and is what every other experiment in this package
        does.
    n_init : int
        Number of k-means++ restarts inside :class:`sklearn.cluster.KMeans`.
        The headline runs use 1; the comparison and ablation blocks use 10.
    max_iter, tol, eta, convergence :
        Iteration cap, stopping tolerance, distance stabiliser and the
        stopping rule.  ``prototype_frobenius`` compares the Frobenius norm of
        the prototype change against ``tol``; ``prototype_maxshift`` compares
        the largest per-cluster prototype shift.
    return_diagnostics : bool
        Also return ``(V, n_iter, init_time, opt_time)``.

    Returns
    -------
    U : ndarray, shape (n_samples, n_clusters)
        Final membership matrix.
    """
    m_i = np.asarray(m_i, dtype=float)
    w_i = np.asarray(w_i, dtype=float)

    if np.any(m_i <= 1.0):
        raise ValueError("m_i must remain greater than 1.")

    if convergence not in CONVERGENCE_CRITERIA:
        raise ValueError(
            "Unknown convergence criterion {!r}".format(convergence)
        )

    n_samples = X.shape[0]

    # ---- initialisation ------------------------------------------------
    t_init_start = time.perf_counter()
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        random_state=random_state,
    )
    V = kmeans.fit(X).cluster_centers_.copy()
    init_time = time.perf_counter() - t_init_start

    # ---- alternating optimisation --------------------------------------
    t_opt_start = time.perf_counter()
    U = np.zeros((n_samples, n_clusters), dtype=float)
    iteration = 0

    for iteration in range(max_iter):
        dist_sq = np.zeros((n_samples, n_clusters), dtype=float)
        for j in range(n_clusters):
            dist_sq[:, j] = np.sum((X - V[j]) ** 2, axis=1) + eta

        U = _membership_update_from_squared_distances(
            dist_sq, m_i, n_samples, n_clusters
        )

        V_new = _prototype_update(X, U, m_i, w_i, n_clusters)

        if convergence == PROTOTYPE_FROBENIUS:
            shift = np.linalg.norm(V_new - V)
        else:
            shift = np.max(np.linalg.norm(V_new - V, axis=1))

        V = V_new

        if shift < tol:
            break

    optimization_time = time.perf_counter() - t_opt_start

    if return_diagnostics:
        return U, V, iteration + 1, init_time, optimization_time
    return U


def fcm_weighted_random_membership(
    X,
    n_clusters,
    m_i,
    w_i,
    random_state,
    max_iter=MAX_ITER,
    tol=1e-5,
    eta=1e-12,
    return_diagnostics=False,
):
    """Weighted fuzzy c-means with random membership initialisation.

    Reproduces the solver used by the m_min / m_max sweeps.  Differences from
    :func:`fcm_weighted` that matter for reproduction:

    * the prototypes start at zero and the first step is a prototype update
      driven by the random memberships, not a k-means++ initialisation;
    * distances are Euclidean rather than squared;
    * the stopping rule compares memberships, ``||U_new - U|| < tol``.
    """
    m_i = np.asarray(m_i, dtype=float)
    w_i = np.asarray(w_i, dtype=float)

    if np.any(m_i <= 1.0):
        raise ValueError("m_i must remain greater than 1.")

    n_samples = X.shape[0]

    t_init_start = time.perf_counter()
    rng = np.random.RandomState(random_state)
    U = rng.rand(n_samples, n_clusters)
    U = U / (np.sum(U, axis=1, keepdims=True) + 1e-12)
    V = np.zeros((n_clusters, X.shape[1]), dtype=float)
    init_time = time.perf_counter() - t_init_start

    t_opt_start = time.perf_counter()
    iteration = 0

    for iteration in range(max_iter):
        # Prototype update first, from the current memberships.
        for j in range(n_clusters):
            weighted = w_i * (U[:, j] ** m_i)
            V[j] = (
                (weighted[:, None] * X).sum(axis=0)
                / (weighted.sum() + 1e-12)
            )

        # Euclidean distances.
        dist = np.zeros((n_samples, n_clusters), dtype=float)
        for j in range(n_clusters):
            dist[:, j] = np.linalg.norm(X - V[j], axis=1) + eta

        U_new = np.zeros_like(U)
        for i in range(n_samples):
            inv_dist = dist[i] ** (-1.0 / (m_i[i] - 1.0))
            U_new[i] = inv_dist / (np.sum(inv_dist) + 1e-12)

        shift = np.linalg.norm(U_new - U)
        U = U_new

        if shift < tol:
            break

    optimization_time = time.perf_counter() - t_opt_start

    if return_diagnostics:
        return U, V, iteration + 1, init_time, optimization_time
    return U
