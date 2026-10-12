"""
Density-to-fuzziness mappings.

All three mappings share the same direction: a high normalised density
(rho_bar -> 1, i.e. a core sample) gives a fuzzifier close to ``m_min``, so
core samples are assigned comparatively crisp memberships, while a low
normalised density (rho_bar -> 0, i.e. a peripheral or contaminating sample)
gives a fuzzifier close to ``m_max``, so those samples are assigned
comparatively fuzzy memberships.

The original notebook declared these functions with two different argument
orders.  This module uses one consistent order and every call site in the
package passes the constants by keyword.
"""

import numpy as np

from .config import ALPHA, BETA, GAMMA, M_MAX, M_MIN


def map_exp(rho_bar, m_min=M_MIN, m_max=M_MAX, alpha=ALPHA):
    """Exponential mapping: m = m_min + (m_max - m_min) * exp(-alpha * rho)."""
    return m_min + (m_max - m_min) * np.exp(-alpha * rho_bar)


def map_rec(rho_bar, m_min=M_MIN, m_max=M_MAX, beta=BETA):
    """Reciprocal mapping: m = m_min + (m_max - m_min) / (1 + beta * rho)."""
    return m_min + (m_max - m_min) / (1.0 + beta * rho_bar)


def map_sig(rho_bar, m_min=M_MIN, m_max=M_MAX, gamma=GAMMA):
    """Sigmoid mapping centred at rho_bar = 0.5."""
    sig = 1.0 / (1.0 + np.exp(-gamma * (rho_bar - 0.5)))
    return m_min + (m_max - m_min) * (1.0 - sig)


MAPPING_FUNCTIONS = {
    "Exp": map_exp,
    "Rec": map_rec,
    "Sig": map_sig,
}


def get_mapping(name):
    """Return the mapping function registered under ``name``."""
    try:
        return MAPPING_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            "Unknown mapping {!r}; expected one of {}".format(
                name, sorted(MAPPING_FUNCTIONS)
            )
        )


def fuzzifier_values(name, rho_bar, m_min=M_MIN, m_max=M_MAX,
                     alpha=ALPHA, beta=BETA, gamma=GAMMA):
    """Evaluate a named mapping on a normalised density vector."""
    function = get_mapping(name)
    if name == "Exp":
        return function(rho_bar, m_min=m_min, m_max=m_max, alpha=alpha)
    if name == "Rec":
        return function(rho_bar, m_min=m_min, m_max=m_max, beta=beta)
    return function(rho_bar, m_min=m_min, m_max=m_max, gamma=gamma)
