# Parameter register

Complete parameter settings, random seeds, parameter-search ranges and the
selection rule for every reported experiment. Contamination ratios are stated
together with their denominator, because two different denominators are used.

The authoritative definition is `ldamfcm/config.py`.

---

## A. Benchmarks and contamination protocol

| Benchmark | Clean n / p / C | Injected contamination | Augmented n | Ratio (denominator) |
|---|---|---|---|---|
| Breast Cancer WDBC | 569 / 30 / 2 | 30 points, uniform between the per-feature min and max of the clean data | 599 | 30/599 = 5.0 % (augmented) |
| Iris | 150 / 4 / 3 | 20 points, uniform on [1.5·min, 1.5·max] per feature | 170 | 20/170 = 11.8 % (augmented) |
| Two-Moons | 300 / 2 / 2 (`noise=0.1`, `random_state=42`) | 40 points, N(0, 1.5²) | 340 | 40/340 = 11.8 % (augmented) |
| Blobs | 400 / 2 / 3 (`cluster_std=[0.6, 1.0, 1.4]`, `random_state=1`) | 60 points, U(-6, 6)² | 460 | 60/460 = 13.0 % (augmented) |
| Anisotropic | 300 / 2 / 3 (`random_state=170`, linear map `[[0.6,-0.6],[-0.4,0.8]]`) | 50 points, uniform between per-feature min and max | 350 | 50/350 = 14.3 % (augmented) |

* Contaminated-point labels: `y = -1`. They are excluded from ARI, DBI and
  cluster recall. Density estimation, weighting and the membership updates are
  computed on the full augmented matrix.
* Preprocessing: one `StandardScaler` per benchmark, fitted on the augmented
  matrix (clean plus injected points) and applied before density estimation.
  The ordering is identical in every experiment.
* **Contamination-ratio denominator — two conventions, not interchangeable:**
  * fixed-size benchmarks: augmented sample count,
    `n_noise / (n_clean + n_noise)`;
  * contamination-ratio sweep (`exp09`): clean sample count,
    `n_noise = floor(ratio · n_clean)`. For Iris this gives 15 / 22 / 30
    injected points at 10 / 15 / 20 %; for Breast 56 / 85 / 113.
* Noise-count stress test (`exp02`): the count is held constant, so the ratio
  over the augmented matrix runs from 6.6 % (40 points) to 22.0 % (160 points).

---

## B. Density model

| Item | Setting |
|---|---|
| Neighbourhood size | k = 12; exact Euclidean kNN, self-distance dropped |
| Bandwidth | σ̃ = global median of the retained kNN distances (one scalar); σ = max{σ̃, 10⁻¹²}, exactly as defined in the manuscript |
| Raw density | ρ_i = (1/k) · Σ_j exp(−d_ij² / σ²) over the k nearest neighbours of i |
| Normalised density | ρ̄_i = (ρ_i − min ρ) / (max ρ − min ρ + 1e-12) |
| Sample weights | w_i = ρ_i / Σ_j ρ_j, denominator guarded by 1e-12 |
| Mapping direction | ρ̄ = 1 (core) → m_i → m_min, crisp; ρ̄ = 0 (periphery) → m_i → m_max, fuzzy. Identical for all three mappings |
| Legacy estimator (historical reproduction only) | σ_i = mean distance to the k neighbours, ρ_i = Σ_j exp(−d_ij² / 2σ_i²), i.e. per-point bandwidth and a sum of kernels. This does **not** match the manuscript definition and is used only by `exp05 --legacy-variant`. It is never used by a default experiment |

---

## C. LD-AMFCM: fixed constants and solver

| Item | Setting |
|---|---|
| Fuzzifier bounds | m_min = 1.05, m_max = 3.5, sweep endpoints taken inclusive |
| Mapping constants | α = 6 (exponential), β = 8 (reciprocal), γ = 12 (sigmoid) |
| Distance stabiliser | η = 1e-9 added to every squared distance; membership denominator guarded by 1e-12 |
| Membership update | u_ik ∝ d_ik^(−1/(m_i−1)), with the sample-wise m_i used both in the exponent and in the prototype update |
| Prototype update | v_k = Σ_i w_i u_ik^(m_i) x_i / Σ_i w_i u_ik^(m_i), denominator guarded by 1e-12 |
| Stopping rule | ‖V_new − V‖_F < 1e-4 in the main runs; max_k ‖v_k^new − v_k‖ < 1e-4 in the scalability runs; ‖U_new − U‖_F < 1e-5 in the sweep variant |
| Iteration cap | 100 (LD-AMFCM and every baseline that reports one) |
| Initialisation | k-means++ in the input space, `random_state = init seed`; identical initial prototypes for FCM, DW-FCM, the three LD-AMFCM mappings and MC-FCM within a matched run. The number of k-means++ restarts differs by experiment family — see the table below |
| Hard labels | ŷ_i = argmax_k u_ik |

### Solver settings that differ by experiment family

Two settings are not global. Both are registered per experiment in
`config.py`.

| Experiment | k-means++ restarts | Membership update |
|---|---|---|
| `exp01` mapping sensitivity | 1 | negative-power |
| `exp02` noise-count stress | 1 | negative-power |
| `exp03` scalability | 1 | negative-power |
| `exp04` neighbourhood size k | 1 | negative-power |
| `exp05` fuzzifier bounds (default, manuscript configuration) | 1 | ratio |
| `exp05` fuzzifier bounds (`--legacy-variant`) | not used | negative-power |
| `exp06` baseline comparison | 10 | ratio |
| `exp07` density-reliability stress | 10 | ratio |
| `exp08` ablation | 10 | ratio |
| `exp09` contamination ratio | 10 | ratio |
| `exp10` runtime | 10 | ratio |
| `exp11` paired statistics | 10 | ratio |

Only the membership column moves the published sweep values. Measured on the
mapping sweep, one and ten restarts give identical ARI at every grid point
tested, so the restarts column restores the recorded protocol rather than
changing any number. The GMM, k-means and adaptive-FCM-GE baselines use ten
restarts in every experiment.

**Membership update forms.** Both are algebraically equivalent, but they
differ wherever the fuzzifier is small, because the negative-power form's
additive 1e-12 guard stops being a rounding protector and becomes the dominant
term of the denominator, so affected rows no longer sum to one:

* *ratio* — `u_ij = 1 / sum_k (d_ij / d_ik) ** (1 / (m_i - 1))`; every row is
  normalised by construction.
* *negative-power* — `inv = d ** (-1 / (m_i - 1)); u = inv / (sum(inv) + 1e-12)`;
  the form the sweep family was produced with.

See `KNOWN_VARIATIONS.md` §14, which also gives the affected values.

---

## D. Candidate grids, search ranges and selection

| Item | Setting |
|---|---|
| α grid | `linspace(2, 15, 10)`: 2.00, 3.44, 4.89, 6.33, 7.78, 9.22, 10.67, 12.11, 13.56, 15.00 |
| β grid | `linspace(2, 20, 10)`: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 |
| γ grid | `linspace(4, 30, 10)`: 4.00, 6.89, 9.78, 12.67, 15.56, 18.44, 21.33, 24.22, 27.11, 30.00 |
| Neighbourhood grid | k ∈ {5, 8, 12, 16, 20, 30} |
| m_min grid | {1.02, 1.05, 1.10, 1.20, 1.30, 1.50} |
| m_max grid | {2.0, 2.5, 3.0, 3.5, 4.0, 5.0} |
| Contamination grids | noise counts {40, 60, 80, 100, 120, 140, 160}; ratios {10, 15, 20} % of n_clean |
| Scalability grids | n ∈ {500, 1000, 2000, 5000, 10000} at p = 10; p ∈ {10, 20, 50, 100, 200, 500} at n = 2000; C = 3; clean data, two informative dimensions plus p − 2 pure-noise dimensions |
| **Selection rule** | The headline configuration (k = 12, α = 6, β = 8, γ = 12, m_min = 1.05, m_max = 3.5) is fixed a priori. The sweeps are sensitivity analyses and never feed back into the reported configuration; every grid point uses one deterministic configuration per run index |
| DBSCAN tuning (disclosed exception) | eps ∈ {0.1, 0.2, …, 2.0} × min_samples ∈ {3, …, 11}; the configuration maximising ARI on the clean ground truth is reported, i.e. a reference-tuned upper bound rather than an untuned competitor |

---

## E. Seeds

See [`SEED_REGISTER.md`](SEED_REGISTER.md) for the full table and the notes on
the offset and coupled designs.

Two randomness sources are separated: a data seed fixing the augmented matrix
and an initialisation seed fixing the initial prototypes.

---

## F. Comparison methods and their settings

| Method | Setting |
|---|---|
| FCM (baseline) | m = 2 constant, w_i = 1, same solver, matched initialisation |
| DW-FCM | m = 2 constant, w_i = ρ_i / Σ_j ρ_j, matched initialisation |
| FCM + m_i | m_i from the sigmoid mapping with w_i = 1 (ablation, noise-count stress) or from the exponential mapping (density-reliability stress) |
| MC-FCM | m_L = 1.05, m_U = 3.5, mapping exponent 1.0, q = max(1, floor(n/C)) nearest distances summed then min–max normalised; no sample weighting |
| AWDFCM | α = 4.0, β = 1.2, θ = 2.0; γ = 0.003 for p ≤ 10 and 0.03 for p > 10; η = 0.5 for p ≤ 6 and 7.0 for p > 6; ε = 0.5·M_X, the source paper prescribes no universal external value; max_iter = 100, tol = 1e-4; centre re-initialisation when ‖v_i − v_j‖ < 1e-3 |
| Adaptive FCM-GE | λ = 1.0, graph k = 10, σ = 2.0, embedding dimension = C, max_iter = 80, tol = 1e-4, γ clipped to [1e-8, 1e8] |
| EM-IFCM | m = 2.0, δ′ = 0.1, max_iter = 200, tol = 1e-6, random normalised membership initialisation |
| k-means++ | n_init = 10, `random_state = init seed` |
| GMM | n_components = C, library-default covariance, `random_state = init seed` |
| Agglomerative | n_clusters = C, library-default linkage; deterministic, dispersion reported as zero |

---

## G. Metrics and evaluation protocol

| Metric | Definition |
|---|---|
| ARI | computed on the uncontaminated samples only, from the identical partition used by every method |
| DBI | standard Davies-Bouldin on the uncontaminated samples; NaN when fewer than two predicted clusters survive there. `exp07` uses the raw form, whose degenerate value is 0 rather than NaN, matching that experiment's original reporting |
| Noise membership | mean over the injected points of max_k u_ik; lower is better |
| Noise-detection AUC | ROC-AUC of (1 − max_k u_ik) against the injected-point indicator, over all samples |
| Density AUC | ROC-AUC of (−ρ_i) against the injected-point indicator; independent of the clustering algorithm |
| Cluster recall | Hungarian assignment of predicted to true cluster labels, then per-cluster recall; minimum and macro average reported |
| Aggregation | mean over the repetitions; dispersion is the sample standard deviation (ddof = 1); deterministic methods reported with zero dispersion |

---

## H. Statistical protocol

| Item | Setting |
|---|---|
| Paired test | paired t-test against the FCM baseline on matched runs |
| Effect size | Cohen's d_z on paired differences, magnitude bands 0.2 / 0.5 / 0.8 |
| Multiplicity | Holm step-down correction |
| Omnibus test | Friedman test on the dataset-level mean ARI (5 datasets × 5 methods), with Kendall's W = χ² / (n_datasets · (k − 1)) |
| Post-hoc | Wilcoxon signed-rank against FCM, Holm-corrected across the four comparisons |
| Significance level | 0.05 |

---

## I. Environment and timing

| Item | Setting |
|---|---|
| Software | Python 3.7.0, NumPy 1.15.1, SciPy 1.1.0, scikit-learn 0.19.2, pandas 0.23.4, Matplotlib 2.2.3, Seaborn 0.9.0. Recorded per run in `results/<experiment>/environment.json` |
| Hardware | Windows 10, AMD64, 16 logical CPUs. Recorded per run in the same file |
| Timing units and scope | wall-clock seconds on one machine. `exp03` uses `time.perf_counter()` and reports density-estimation, initialisation and optimisation time separately, with exact kNN search (`algorithm="brute"`, `n_jobs=-1`). `exp10` uses `time.time()` and reports preprocessing plus iteration time on the contaminated Breast benchmark. Only relative timings are interpreted |

---

## J. Frozen outputs

| Category | Files |
|---|---|
| Per-run tables | `strong_fcm_baselines_per_run.csv`, `noise_count_stress_raw.csv`, `scalability_{sample_size,dimensionality}_raw.csv`, `density_reliability_stress_raw.csv`, `paired_runs.csv` |
| Aggregated summaries | `strong_fcm_baselines_results.csv`, `strong_fcm_baselines_overall_summary.csv`, `strong_fcm_baselines_results_mean_std.csv`, `strong_fcm_baselines_overall_summary_mean_std.csv`, `noise_count_stress_summary.csv`, `scalability_*_summary.csv`, `density_reliability_stress_summary.csv`, `density_diagnostics_summary.csv`, `ablation_results.csv`, `mean_std_summary.csv`, `dataset_level_mean_ari.csv` |
| Statistical tables | `paired_ttest_holm.csv`, `friedman_test.csv`, `wilcoxon_posthoc.csv` |
| Figures | sensitivity, k-sensitivity, fuzzifier, scalability, stress and comparison figures under `figures/<experiment>/` |
