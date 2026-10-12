"""
Comparison methods that are not plain weighted fuzzy c-means.

Each of these follows its source publication.  Where the source paper does not
fix a value for an external benchmark, the convention actually used in the
original notebook is reproduced here and documented in the docstring, because
the published comparison table was produced with it.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .config import (
    AFCM_GE_K,
    AFCM_GE_LAMBDA,
    AFCM_GE_MAX_ITER,
    AFCM_GE_SIGMA,
    AFCM_GE_TOL,
    AWDFCM_ALPHA,
    AWDFCM_BETA,
    AWDFCM_ETA_HIGH_DIM,
    AWDFCM_ETA_LOW_DIM,
    AWDFCM_GAMMA_HIGH_DIM,
    AWDFCM_GAMMA_LOW_DIM,
    AWDFCM_MAX_ITER,
    AWDFCM_REINIT_THRESHOLD,
    AWDFCM_THETA,
    AWDFCM_TOL,
    EM_IFCM_DELTA_PRIME,
    EM_IFCM_M,
    EM_IFCM_MAX_ITER,
    EM_IFCM_TOL,
    KMEANS_N_INIT,
    MC_FCM_MAPPING_EXPONENT,
    MC_FCM_M_LOW,
    MC_FCM_M_UP,
)


# ======================================================================
# MC-FCM : sample-specific fuzzification coefficients
# ======================================================================

def mc_fcm_fuzzifier(X, n_clusters,
                     m_low=MC_FCM_M_LOW,
                     m_up=MC_FCM_M_UP,
                     exponent=MC_FCM_MAPPING_EXPONENT):
    """Sample-specific fuzzification coefficients (Khang et al., 2020).

    For each sample the sum of the distances to the ``floor(n / C)`` closest
    samples is taken as the concentration measure ``delta_star``, min-max
    normalised and mapped linearly onto ``[m_low, m_up]``.  A higher
    concentration gives a higher fuzzifier, i.e. the direction is the same as
    the LD-AMFCM mappings: dense regions are treated as fuzzy and sparse
    regions as crisp.
    """
    n_samples = X.shape[0]

    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    q = max(1, n_samples // n_clusters)

    closest_distances = np.partition(D, q - 1, axis=1)[:, :q]
    delta_star = closest_distances.sum(axis=1)

    delta_min = delta_star.min()
    delta_max = delta_star.max()

    if delta_max - delta_min < 1e-12:
        normalized = np.full(n_samples, 0.5, dtype=float)
    else:
        normalized = (delta_star - delta_min) / (delta_max - delta_min)

    return m_low + (m_up - m_low) * (normalized ** exponent)


# ======================================================================
# AWDFCM : adaptive weighted data fuzzy c-means
# ======================================================================

def awdfcm_initial_centers(X, n_clusters, alpha=AWDFCM_ALPHA, beta=AWDFCM_BETA,
                           random_state=42):
    """AWDFCM centre initialisation (Wu et al., Neurocomputing 552, 2023)."""
    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)

    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    d_k = D.mean(axis=1)
    mean_diversity = d_k.mean()

    radius = mean_diversity / alpha
    min_center_distance = beta * mean_diversity

    neighborhood_counts = (D <= radius).sum(axis=1)

    selected = []
    available = np.ones(n_samples, dtype=bool)

    while len(selected) < n_clusters:
        candidates = np.where(available)[0]
        if len(candidates) == 0:
            break

        order = candidates[np.argsort(-neighborhood_counts[candidates])]
        chosen = None

        for idx in order:
            if len(selected) == 0:
                chosen = idx
                break
            if np.all(D[idx, selected] > min_center_distance):
                chosen = idx
                break

        if chosen is None:
            # The published separation threshold can be too strict for an
            # external benchmark; fall back to the maximum-minimum rule.
            remaining = [idx for idx in candidates]
            if len(selected) > 0:
                chosen = max(remaining, key=lambda idx: np.min(D[idx, selected]))
            else:
                chosen = remaining[0]

        selected.append(chosen)
        available[chosen] = False
        neighborhood_counts[chosen] = -1

    while len(selected) < n_clusters:
        remaining = np.where(available)[0]
        if len(remaining) == 0:
            selected.append(rng.randint(0, n_samples))
        elif len(selected) == 0:
            selected.append(remaining[0])
            available[remaining[0]] = False
        else:
            chosen = max(remaining, key=lambda idx: np.min(D[idx, selected]))
            selected.append(chosen)
            available[chosen] = False

    return X[np.array(selected[:n_clusters])].copy(), d_k, mean_diversity


def awdfcm_data_weights(d_k, mean_diversity, epsilon=None):
    """AWDFCM data-weighting rule.

    The source paper defines epsilon as a preset positive threshold without
    giving one universal value for arbitrary external benchmarks, so the
    label-free convention ``epsilon = 0.5 * M_X`` is used, exactly as in the
    notebook.
    """
    if epsilon is None:
        epsilon = 0.5 * mean_diversity

    if epsilon >= mean_diversity:
        epsilon = 0.5 * mean_diversity

    weights = np.ones_like(d_k, dtype=float)
    mask = d_k > epsilon
    weights[mask] = np.exp(
        -(d_k[mask] - epsilon) / (mean_diversity - epsilon + 1e-12)
    )
    return weights, epsilon


def awdfcm(X, n_clusters, theta=AWDFCM_THETA, gamma=AWDFCM_GAMMA_LOW_DIM,
           eta=None, alpha=AWDFCM_ALPHA, beta=AWDFCM_BETA, epsilon=None,
           max_iter=AWDFCM_MAX_ITER, tol=AWDFCM_TOL, random_state=42,
           reinit_mode="fresh_rng"):
    """AWDFCM objective and alternating updates.

    ``reinit_mode`` reproduces the two forms that appear in the notebook:
    ``"fresh_rng"`` constructs a new ``RandomState(random_state)`` for every
    re-initialisation, so the replacement point is deterministic (this is what
    the reported ten-run table used); ``"global"`` draws from the global NumPy
    random state.
    """
    n_samples, n_features = X.shape

    if eta is None:
        eta = AWDFCM_ETA_LOW_DIM if n_features <= 6 else AWDFCM_ETA_HIGH_DIM

    V, d_k, mean_diversity = awdfcm_initial_centers(
        X, n_clusters, alpha=alpha, beta=beta, random_state=random_state
    )
    W, _epsilon_used = awdfcm_data_weights(d_k, mean_diversity, epsilon=epsilon)

    F = np.ones(n_features, dtype=float) / n_features
    delta = np.zeros(n_clusters, dtype=float)

    # Initial membership from the current centres.
    dist0 = np.zeros((n_samples, n_clusters), dtype=float)
    for i in range(n_samples):
        for j in range(n_clusters):
            dist0[i, j] = np.sum((X[i] - V[j]) ** 2) + 1e-12

    U = np.zeros((n_samples, n_clusters), dtype=float)
    for i in range(n_samples):
        inv = 1.0 / dist0[i]
        U[i] = inv / (inv.sum() + 1e-12)

    for _ in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()
        F_old = F.copy()
        delta_old = delta.copy()

        # ---- Step 1: membership update --------------------------------
        dist_fw = np.zeros((n_samples, n_clusters), dtype=float)
        for i in range(n_samples):
            for j in range(n_clusters):
                dist_fw[i, j] = np.sum(F * (X[i] - V[j]) ** 2) + 1e-12

        reg = gamma / (np.maximum(1.0 + delta, 1e-12) ** (n_features * n_clusters))

        denom_terms = W[:, None] * dist_fw + reg[None, :]
        denom_terms = np.maximum(denom_terms, 1e-14)
        inv_terms = 1.0 / denom_terms
        U = inv_terms / (inv_terms.sum(axis=1, keepdims=True) + 1e-12)

        # ---- Step 2: feature-weight update ----------------------------
        scatter = np.zeros(n_features, dtype=float)
        for j in range(n_features):
            total = 0.0
            for i in range(n_clusters):
                total += np.sum(W * (U[:, i] ** 2) * (X[:, j] - V[i, j]) ** 2)
            scatter[j] = total

        exponent = -scatter / max(eta, 1e-12)
        exponent -= np.max(exponent)
        ef = np.exp(exponent)
        F = ef / (ef.sum() + 1e-12)

        # ---- Step 3: centre update ------------------------------------
        V_new = np.zeros_like(V)
        for i in range(n_clusters):
            weights = W * (U[:, i] ** 2)
            denom = weights.sum()
            if denom <= 1e-12:
                V_new[i] = V[i]
            else:
                V_new[i] = (weights[:, None] * X).sum(axis=0) / denom
        V = V_new

        # ---- Step 4: adaptive parameter update ------------------------
        labels = np.argmax(U, axis=1)
        delta = np.zeros(n_clusters, dtype=float)
        for i in range(n_clusters):
            idx = np.where(labels == i)[0]
            if len(idx) == 0:
                delta[i] = delta_old[i]
                continue
            mean_i = np.mean(F[None, :] * X[idx], axis=0)
            weighted_dist = np.sqrt(
                np.sum(F[None, :] * (X[idx] - mean_i) ** 2, axis=1)
            )
            delta[i] = np.mean(weighted_dist)

        # ---- centre re-initialisation check ---------------------------
        if n_clusters > 1:
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    if np.linalg.norm(V[i] - V[j]) < AWDFCM_REINIT_THRESHOLD:
                        if reinit_mode == "fresh_rng":
                            idx = np.random.RandomState(random_state).randint(
                                0, n_samples
                            )
                        else:
                            idx = np.random.randint(0, n_samples)
                        V[j] = X[idx].copy()

        change = max(
            np.linalg.norm(U - U_old),
            np.linalg.norm(V - V_old),
            np.linalg.norm(F - F_old),
            np.linalg.norm(delta - delta_old),
        )
        if change < tol:
            break

    return U


# ======================================================================
# Adaptive FCM with graph embedding
# ======================================================================

def adaptive_fcm_ge(X, n_clusters, lam=AFCM_GE_LAMBDA, k=AFCM_GE_K,
                    sigma=AFCM_GE_SIGMA, max_iter=AFCM_GE_MAX_ITER,
                    tol=AFCM_GE_TOL, random_state=42):
    """Adaptive FCM with graph embedding.

    One-stage alternating optimisation of the embedding, the prototypes and
    the entropy-softmax memberships.  The embedding dimension is the number of
    clusters.
    """
    n_samples, _ = X.shape

    k_graph = min(k, n_samples - 1)
    nbrs = NearestNeighbors(n_neighbors=k_graph + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    W = np.zeros((n_samples, n_samples), dtype=float)
    for i in range(n_samples):
        for r in range(k_graph):
            j = indices[i, r]
            sqdist = np.sum((X[i] - X[j]) ** 2)
            W[i, j] = np.exp(-sqdist / (2.0 * sigma ** 2))
    W = 0.5 * (W + W.T)

    degree = W.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-12))
    laplacian = np.diag(degree) - W
    laplacian_hat = d_inv_sqrt.dot(laplacian).dot(d_inv_sqrt)

    km = KMeans(n_clusters=n_clusters, n_init=KMEANS_N_INIT,
                random_state=random_state).fit(X)
    V0 = km.cluster_centers_

    dist0 = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):
        for j in range(n_clusters):
            dist0[i, j] = np.sum((X[i] - V0[j]) ** 2) + 1e-12

    gamma = 1.0
    logits = -gamma * dist0
    logits -= logits.max(axis=1, keepdims=True)
    E = np.exp(logits)
    U = E / (E.sum(axis=1, keepdims=True) + 1e-12)

    d_tilde = n_clusters
    X_tilde = np.zeros((n_samples, d_tilde))
    V = np.zeros((n_clusters, d_tilde))

    for _ in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()

        col_sums = U.sum(axis=0)
        B = np.diag(1.0 / (col_sums + 1e-12))
        M = np.eye(n_samples) - U.dot(B).dot(U.T)

        A = gamma * M + lam * laplacian_hat
        A = 0.5 * (A + A.T)

        try:
            _eigvals, eigvecs = np.linalg.eigh(A)
        except np.linalg.LinAlgError:
            break

        X_tilde = eigvecs[:, :d_tilde]

        for j in range(n_clusters):
            weights = U[:, j]
            denom = weights.sum() + 1e-12
            V[j] = (weights[:, None] * X_tilde).sum(axis=0) / denom

        projected_dist = np.zeros((n_samples, n_clusters))
        for i in range(n_samples):
            for j in range(n_clusters):
                projected_dist[i, j] = (
                    np.sum((X_tilde[i] - V[j]) ** 2) + 1e-12
                )

        denominator = np.sum(U * projected_dist)
        gamma = 0.5 * d_tilde * n_samples / (denominator + 1e-12)
        gamma = np.clip(gamma, 1e-8, 1e8)

        logits = -gamma * projected_dist
        logits -= logits.max(axis=1, keepdims=True)
        E = np.exp(logits)
        U = E / (E.sum(axis=1, keepdims=True) + 1e-12)

        if max(np.linalg.norm(U - U_old), np.linalg.norm(V - V_old)) < tol:
            break

    return U


# ======================================================================
# EM-IFCM
# ======================================================================

def em_ifcm(X, n_clusters, m=EM_IFCM_M, delta_prime=EM_IFCM_DELTA_PRIME,
            max_iter=EM_IFCM_MAX_ITER, tol=EM_IFCM_TOL, random_state=42):
    """EM-IFCM with fuzzy sample-size proportions.

    The memberships are initialised at random, so this method is the only
    input-space comparison method whose spread across repetitions is not
    driven purely by the prototype initialisation.
    """
    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)

    U = rng.rand(n_samples, n_clusters)
    U /= U.sum(axis=1, keepdims=True) + 1e-12

    delta_i = np.zeros(n_samples)
    V = np.zeros((n_clusters, X.shape[1]))

    for j in range(n_clusters):
        weights = U[:, j] ** m
        V[j] = (weights[:, None] * X).sum(axis=0) / (weights.sum() + 1e-12)

    for _ in range(max_iter):
        U_old = U.copy()
        V_old = V.copy()

        for j in range(n_clusters):
            weights = U[:, j] ** m
            V[j] = (weights[:, None] * X).sum(axis=0) / (weights.sum() + 1e-12)

        fuzzy_size = U.sum(axis=0)
        f = fuzzy_size / (fuzzy_size.sum() + 1e-12)

        dist_sq = np.zeros((n_samples, n_clusters))
        for i in range(n_samples):
            for j in range(n_clusters):
                dist_sq[i, j] = np.sum((X[i] - V_old[j]) ** 2) + 1e-12

        D_star = dist_sq / (dist_sq.sum(axis=1, keepdims=True) + 1e-12)

        delta_new = np.zeros(n_samples)
        for i in range(n_samples):
            inner_sum = 0.0
            for j in range(n_clusters):
                numerator = (1.0 - delta_prime * D_star[i, j]) * f[j]
                denominator = m * (
                    dist_sq[i, j] - f[j] * delta_i[i] * D_star[i, j]
                )
                denominator = max(denominator, 1e-12)
                ratio = max(numerator / denominator, 1e-12)
                inner_sum += ratio ** (1.0 / (m - 1.0))
            delta_new[i] = delta_prime * max(inner_sum, 1e-12) ** (1.0 - m)
        delta_i = delta_new

        U_new = np.zeros((n_samples, n_clusters))
        for i in range(n_samples):
            terms = np.zeros(n_clusters)
            for j in range(n_clusters):
                numerator = (1.0 - delta_prime * D_star[i, j]) * f[j]
                effective_distance = (
                    dist_sq[i, j] - f[j] * delta_i[i] * D_star[i, j]
                )
                effective_distance = max(effective_distance, 1e-12)
                terms[j] = max(numerator / effective_distance, 1e-12)
            transformed = terms ** (1.0 / (m - 1.0))
            U_new[i] = transformed / (transformed.sum() + 1e-12)

        U = U_new

        if max(np.linalg.norm(U - U_old), np.linalg.norm(V - V_old)) < tol:
            break

    return U


# ======================================================================
# Dimensionality-dependent AWDFCM settings
# ======================================================================

def awdfcm_dimensional_settings(n_features):
    """Return ``(gamma, eta)`` for the AWDFCM published low/high-dim grouping.

    gamma = 0.003 for p <= 10 and 0.03 for p > 10;
    eta   = 0.5   for p <= 6  and 7.0  for p > 6.
    """
    gamma = (
        AWDFCM_GAMMA_LOW_DIM if n_features <= 10 else AWDFCM_GAMMA_HIGH_DIM
    )
    eta = AWDFCM_ETA_LOW_DIM if n_features <= 6 else AWDFCM_ETA_HIGH_DIM
    return gamma, eta
