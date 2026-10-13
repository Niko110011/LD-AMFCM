"""
Central parameter register for the LD-AMFCM reproducibility package.

Every value in this module is taken verbatim from the original experimental
notebook.  Nothing here is chosen by the authors of this package.  If a value
is changed, the published numbers will change with it.

The two randomness sources are kept strictly separate:

  * a *data seed* fixes the augmented benchmark matrix, i.e. the base samples
    plus the injected contamination;
  * an *initialisation seed* fixes the prototype initialisation of a method.

A matching numeric value for the two does not imply matching inputs or
matching initial prototypes, so they are declared and reported separately.
"""

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# LD-AMFCM fixed constants (headline configuration reported in the paper)
# ----------------------------------------------------------------------

K_NEIGHBORS = 12        # kNN neighbourhood size for the local density
M_MIN = 1.05            # lower fuzzifier bound
M_MAX = 3.5             # upper fuzzifier bound
ALPHA = 6.0             # exponential mapping constant
BETA = 8.0              # reciprocal mapping constant
GAMMA = 12.0            # sigmoid mapping constant
ETA = 1e-9              # additive stabiliser on squared distances
MAX_ITER = 100          # iteration cap
TOL = 1e-4              # prototype-change tolerance


# ----------------------------------------------------------------------
# Solver variant used by the m_min / m_max sweeps
# ----------------------------------------------------------------------

# The m_min / m_max sweeps in the original notebook were produced with a
# different density estimator, a random membership initialisation and a
# membership-change tolerance instead of a prototype-change tolerance.
# This variant is reproduced exactly as it was run, and is flagged as such in
# every output table.  See docs/KNOWN_VARIATIONS.md.
VARIANT_TOL = 1e-5
VARIANT_DENSITY_ESTIMATOR = "sum_gaussian_perpoint_mean"
VARIANT_INIT = "random_membership"

# Density estimator used by the headline configuration.
MAIN_DENSITY_ESTIMATOR = "mean_gaussian_global_median"
MAIN_INIT = "kmeans++"


# ----------------------------------------------------------------------
# Candidate grids and search ranges
# ----------------------------------------------------------------------

ALPHA_GRID = [float(v) for v in np.linspace(2.0, 15.0, 10)]
BETA_GRID = [float(v) for v in np.linspace(2.0, 20.0, 10)]
GAMMA_GRID = [float(v) for v in np.linspace(4.0, 30.0, 10)]

K_GRID = [5, 8, 12, 16, 20, 30]

M_MIN_GRID = [1.02, 1.05, 1.10, 1.20, 1.30, 1.50]
M_MAX_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

NOISE_COUNT_GRID = [40, 60, 80, 100, 120, 140, 160]
NOISE_RATIO_GRID = [0.10, 0.15, 0.20]

# Scalability study
N_VALUES = [500, 1000, 2000, 5000, 10000]
P_VALUES = [10, 20, 50, 100, 200, 500]
FIXED_P = 10
FIXED_N = 2000
N_CLUSTERS_SCALABILITY = 3
N_REPEATS_SCALABILITY = 5
SCALABILITY_GENERATOR_SEED = 42   # data are held fixed across the repetitions

# DBSCAN reference-tuned upper bound: the reported configuration is the one
# maximising ARI on the clean ground truth, i.e. it is an oracle-tuned
# reference point rather than an untuned competitor.
DBSCAN_EPS_GRID = [round(0.1 * i, 1) for i in range(1, 21)]
DBSCAN_MIN_SAMPLES_GRID = list(range(3, 12))

# Baselines that follow their source publication
MC_FCM_M_LOW = 1.05
MC_FCM_M_UP = 3.5
MC_FCM_MAPPING_EXPONENT = 1.0

AWDFCM_ALPHA = 4.0
AWDFCM_BETA = 1.2
AWDFCM_THETA = 2.0
AWDFCM_GAMMA_LOW_DIM = 0.003     # p <= 10
AWDFCM_GAMMA_HIGH_DIM = 0.03     # p > 10
AWDFCM_ETA_LOW_DIM = 0.5         # p <= 6
AWDFCM_ETA_HIGH_DIM = 7.0        # p > 6
AWDFCM_MAX_ITER = 100
AWDFCM_TOL = 1e-4
AWDFCM_REINIT_THRESHOLD = 0.001

AFCM_GE_LAMBDA = 1.0
AFCM_GE_K = 10
AFCM_GE_SIGMA = 2.0
AFCM_GE_MAX_ITER = 80
AFCM_GE_TOL = 1e-4

EM_IFCM_M = 2.0
EM_IFCM_DELTA_PRIME = 0.1
EM_IFCM_MAX_ITER = 200
EM_IFCM_TOL = 1e-6

KMEANS_N_INIT = 10
GMM_N_COMPONENTS_IS_C = True


# ----------------------------------------------------------------------
# k-means++ restart count, registered per experiment
# ----------------------------------------------------------------------

# The number of k-means++ restarts differs by experiment family in the
# original notebook, so it is registered per experiment rather than globally.
#
#   n_init = 1   mapping, k, scalability and noise-count sweeps
#   n_init = 10  comparison, ablation, stress, noise-ratio, runtime and
#                paired studies
#   not used     m_min / m_max sweeps (random-membership variant)
#
# Note on what this does and does not affect: measured on the mapping sweep,
# 1 and 10 restarts give identical ARI at every grid point tested, so this
# registration restores the recorded protocol rather than changing any
# published number.  The membership form (see MEMBERSHIP_FORMS in fcm.py) is
# the setting that actually moves the sweep results.
N_INIT_BY_EXPERIMENT = {
    "exp01_mapping_sensitivity": 1,
    "exp02_noise_count_stress": 1,
    "exp03_scalability": 1,
    "exp04_k_sensitivity": 1,
    # The default path of exp05 is the random-membership variant and takes no
    # k-means initialisation; the value 1 applies to its --headline-solver
    # path, which follows the sweep-family convention.
    "exp05_fuzzifier_sensitivity": 1,
    "exp06_baseline_comparison": 10,
    "exp07_density_reliability": 10,
    "exp08_ablation": 10,
    "exp09_noise_ratio": 10,
    "exp10_runtime": 10,
    "exp11_paired_statistics": 10,
}


def n_init_for(experiment):
    """Number of k-means++ restarts used by the named experiment."""
    try:
        return N_INIT_BY_EXPERIMENT[experiment]
    except KeyError:
        raise KeyError(
            "No n_init registered for experiment {!r}".format(experiment)
        )


# ----------------------------------------------------------------------
# Seed register
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class SeedSpec:
    """Separates the data seed from the initialisation seed.

    Attributes
    ----------
    data_seed : int
        First value of the data-seed stream.
    init_seed : int
        First value of the initialisation-seed stream.
    n_runs : int
        Number of repetitions; run i uses ``data_seed + i`` and
        ``init_seed + i`` respectively.
    regenerate_data : bool
        ``True``  -> the contaminated matrix is regenerated for every run
                     (regenerated-input protocol).
        ``False`` -> the contaminated matrix is generated once and frozen,
                     and only the initialisation varies (frozen-input protocol).
    description : str
        Human-readable description of where this specification applies.
    """

    data_seed: int
    init_seed: int
    n_runs: int
    regenerate_data: bool
    description: str

    def data_seeds(self):
        return [self.data_seed + i for i in range(self.n_runs)]

    def init_seeds(self):
        return [self.init_seed + i for i in range(self.n_runs)]


# Main comparison: contaminated matrices generated once under the data seed and
# then frozen; the ten repetitions vary the initialisation seed only.
SEEDS_MAIN_COMPARISON = SeedSpec(
    data_seed=42,
    init_seed=0,
    n_runs=10,
    regenerate_data=False,
    description="Baseline comparison, frozen inputs (10 repetitions)",
)

# Noise-count stress test: the two streams are deliberately offset so that the
# data seed and the initialisation seed are not confounded.
SEEDS_NOISE_COUNT = SeedSpec(
    data_seed=100,
    init_seed=0,
    n_runs=10,
    regenerate_data=True,
    description="Noise-count stress test (data seed 100+s, init seed s)",
)

# Density-reliability stress test: both streams carry the same index by
# construction of the original experiment.  This coupling is disclosed.
SEEDS_DENSITY_STRESS = SeedSpec(
    data_seed=42,
    init_seed=42,
    n_runs=10,
    regenerate_data=True,
    description="Density-reliability stress test (coupled seed 42+s)",
)

# Repeated runs with regenerated inputs, reported separately from the
# frozen-input protocol and never pooled with it.
SEEDS_REGENERATED_INPUT = SeedSpec(
    data_seed=0,
    init_seed=0,
    n_runs=10,
    regenerate_data=True,
    description="Regenerated-input repeated study",
)

# Single-run analyses
SEED_SINGLE_RUN = 42


# ----------------------------------------------------------------------
# Benchmark definitions
# ----------------------------------------------------------------------

# name -> (clean samples, features, clusters)
BENCHMARK_SHAPES = {
    "Blobs + Uniform Noise": (400, 2, 3),
    "Iris + Outliers": (150, 4, 3),
    "Moons + Gaussian Noise": (300, 2, 2),
    "Anisotropic + Noise": (300, 2, 3),
    "Breast Cancer + Outliers": (569, 30, 2),
}

# Order in which the benchmarks are generated.  The generators consume one
# shared global NumPy stream, so this order is part of the experimental
# protocol and must not be changed if the published numbers are to be
# reproduced.
BENCHMARK_ORDER = [
    "Blobs + Uniform Noise",
    "Iris + Outliers",
    "Moons + Gaussian Noise",
    "Anisotropic + Noise",
    "Breast Cancer + Outliers",
]

# Contamination ratio denominator conventions.  The two conventions are not
# interchangeable and are reported separately.
#   "augmented" -> n_noise / (n_clean + n_noise)   (fixed-size benchmarks)
#   "clean"     -> n_noise = floor(ratio * n_clean) (noise-ratio sweep)
CONTAMINATION_RATIO_DENOMINATOR_FIXED = "augmented"
CONTAMINATION_RATIO_DENOMINATOR_SWEEP = "clean"

NOISE_LABEL = -1


# ----------------------------------------------------------------------
# Cluster labels used in output tables
# ----------------------------------------------------------------------

METHOD_FCM = "FCM (Baseline)"
METHOD_DW_FCM = "DW-FCM"
METHOD_FCM_MI = "FCM + m_i"
METHOD_LD_EXP = "LD-AMFCM (Exp)"
METHOD_LD_REC = "LD-AMFCM (Rec)"
METHOD_LD_SIG = "LD-AMFCM (Sig)"
METHOD_MC_FCM = "MC-FCM"
METHOD_AWDFCM = "AWDFCM"
METHOD_AFCM_GE = "Adaptive FCM-GE"
METHOD_EM_IFCM = "EM-IFCM"
METHOD_KMEANS = "K-Means++"
METHOD_GMM = "GMM"
METHOD_DBSCAN = "DBSCAN (Opt)"
METHOD_AGGLOMERATIVE = "Agglomerative"

# Order used in the original notebook
STANDARD_METHODS_ORDER = [
    METHOD_FCM,
    METHOD_DW_FCM,
    METHOD_LD_EXP,
    METHOD_LD_REC,
    METHOD_LD_SIG,
    METHOD_MC_FCM,
    METHOD_AWDFCM,
    METHOD_AFCM_GE,
    METHOD_EM_IFCM,
]

MAPPING_NAMES = ["Exp", "Rec", "Sig"]


def mapping_label(mapping_name):
    """Return the column label used for a mapping name."""
    return {
        "Exp": METHOD_LD_EXP,
        "Rec": METHOD_LD_REC,
        "Sig": METHOD_LD_SIG,
    }[mapping_name]
