# LD-AMFCM — reproducibility package

Local-density adaptive fuzzy c-means: data generation, the proposed method and
its parameter sweeps, all comparison methods, the metric definitions and the
statistical protocol, in a form that regenerates the reported numbers.

---

## 1. Quick start

```bash
pip install -r requirements.txt

python smoke_test.py                     # verify against the published values, ~1 min
python verify_published_sweeps.py        # check cross-table consistency, ~1 s
python verify_published_sweeps.py --legacy  # reproduce the notebook's sweeps
python run_all.py --quick                # every experiment on reduced grids, ~1 min
python run_all.py                        # the full suite
```

Results are written to `results/<experiment>/` and figures to
`figures/<experiment>/`. Each experiment drops an `environment.json` next to
its tables recording the Python, NumPy, SciPy, scikit-learn, pandas,
Matplotlib and Seaborn versions, the platform and the CPU count used for that
run.

Individual experiments:

```bash
python experiments/exp06_baseline_comparison.py         # main comparison table
python experiments/exp06_baseline_comparison.py --runs 1  # single-run version
python experiments/exp05_fuzzifier_sensitivity.py --headline-solver
python run_all.py --experiments exp06,exp11
python run_all.py --list
```

---

## 2. Layout

```
ldamfcm/
    config.py        every constant, grid and seed specification
    density.py       the two local-density estimators
    mappings.py      the three density-to-fuzziness mappings
    fcm.py           the two weighted fuzzy c-means solvers
    models.py        MC-FCM, AWDFCM, Adaptive FCM-GE, EM-IFCM
    datasets.py      benchmark and stress-scenario generation
    metrics.py       ARI, DBI, noise diagnostics, matched cluster recall
    stats.py         paired t-test, Cohen's d_z, Holm, Friedman, Wilcoxon
    pipeline.py      shared helper that builds (m_i, w_i) and evaluates
    io_utils.py      result directories, tables, figures, environment report
experiments/
    exp01 … exp11    one script per reported experiment
run_all.py           runs the suite, one process per experiment
smoke_test.py        checks the package against the published ARI values
docs/                parameter register, seed register, known variations
```

`config.py` is the single source of truth. No experiment script invents a
parameter; if a number in `config.py` changes, the published numbers change
with it.

---

## 3. Two randomness sources, kept separate

The package never conflates the two:

| | what it fixes | where it is declared |
|---|---|---|
| **data seed** | the augmented benchmark matrix: base samples plus injected contamination | `config.SeedSpec.data_seed` |
| **initialisation seed** | the initial prototypes of a method | `config.SeedSpec.init_seed` |

Reproducing identical inputs does not follow from reusing the same integer in
both roles, so they are declared, reported and written to the output tables
separately.

### Two input protocols

**Frozen input** (`regenerate_data=False`) — the contaminated matrix of each
benchmark is generated once under the data seed and then held fixed; the ten
repetitions vary the initialisation seed only. Within a matched run, FCM,
DW-FCM, the three LD-AMFCM mappings and MC-FCM all receive the same
initialisation seed on the same matrix and therefore start from **identical
initial prototypes**. This is the main comparison (`exp06`) and the mapping,
`k`, and fuzzifier sweeps.

**Regenerated input** (`regenerate_data=True`) — both the contamination and
the initial prototypes are redrawn for each repetition. This is the noise-count
stress test (`exp02`), the density-reliability scenarios (`exp07`) and the
paired statistics (`exp11`).

The two protocols answer different questions and are **never pooled**.

The seed register, including the offsetting convention that keeps the two
streams unconfounded in `exp02`, is in [`docs/SEED_REGISTER.md`](docs/SEED_REGISTER.md).

---

## 4. Configuration the paper reports

```
k = 12          neighbourhood size
m_min = 1.05    lower fuzzifier bound
m_max = 3.5     upper fuzzifier bound
alpha = 6       exponential mapping
beta  = 8       reciprocal mapping
gamma = 12      sigmoid mapping
eta = 1e-9      additive stabiliser on squared distances
tolerance 1e-4  prototype change
max_iter 100
```

All three mappings run in the same direction: a dense core sample
(`rho_bar -> 1`) receives a fuzzifier near `m_min` and therefore a crisp
membership, while a peripheral sample (`rho_bar -> 0`) receives one near
`m_max` and a fuzzy membership.

Full register, including every candidate grid and search range:
[`docs/PARAMETER_REGISTER.md`](docs/PARAMETER_REGISTER.md).

---

## 5. What the package deliberately preserves

Two configurations appear in this package that are **not** the headline
configuration. They are reproduced because the published numbers were produced
with them, and every table they generate is labelled accordingly.

1. **The `m_min` / `m_max` sweep now follows the manuscript.** It previously
   used a different density estimator (per-point bandwidth, sum of Gaussian
   kernels), which pinned most fuzzifiers at `m_min`, made the density weights
   uninformative and flattened the whole `m_max` table. `exp05` now uses the
   manuscript estimator and the headline solver; its default point matches the
   `k` sweep. The original implementation is behind `--legacy-variant` for
   historical reproduction. See `docs/KNOWN_VARIATIONS.md` §1.

2. **DBSCAN is a reference-tuned upper bound.** Its two hyperparameters are
   searched over a declared grid and the configuration with the best ARI on
   the clean ground truth is reported. It is not an untuned competitor.

3. **The scalability timings measure thread start-up.** `exp03` defaults to
   `n_jobs=-1` for the exact kNN search, as the original did, and in this
   environment that path is dominated by joblib worker-pool start-up —
   3.95 s against 0.005 s at n = 500. The default reproduces the published
   columns; `--n-jobs 1` produces timings that reflect the algorithm. See
   `docs/KNOWN_VARIATIONS.md` §13 before quoting the figure.

4. **The sweep family's original numbers came from a fragile membership
   update.** Two algebraically equivalent forms appear in the notebook, and
   they disagree wherever the fuzzifier is small: the negative-power form's
   additive 1e-12 guard becomes the dominant term of its denominator, so
   affected rows stop summing to one. The original mapping and `k` sweeps were
   produced with it. **Every default experiment now uses the numerically
   stable ratio form**, so all tables share one implementation and agree at the
   shared default point; the negative-power form is available behind
   `--legacy-membership` (and `exp05 --legacy-variant`) for reproducing the
   original curves. See `docs/KNOWN_VARIATIONS.md` §14.

The number of k-means++ restarts also differs by family — 1 restart for the
sweeps, 10 for the comparison and ablation — and is registered per experiment
in `config.py`. See the solver-settings table in
`docs/PARAMETER_REGISTER.md`.

Details and the full list of deviations: [`docs/KNOWN_VARIATIONS.md`](docs/KNOWN_VARIATIONS.md).

---

## 6. Verification

`smoke_test.py` reruns the frozen-input comparison protocol and compares the
resulting ARI against the values saved in the original experiment output:

```bash
python smoke_test.py --benchmark "Breast Cancer + Outliers"
```

All five benchmarks reproduce the published ARI to four decimals for FCM,
DW-FCM, the three LD-AMFCM mappings and MC-FCM. Expected values and the full
comparison table are in [`docs/EXPECTED_RESULTS.md`](docs/EXPECTED_RESULTS.md).

The code was also reformed where the original was silently order-dependent:
the membership update is evaluated as a pairwise distance ratio, and the
convergence test no longer compares a quantity against itself. Both changes
are algebraically neutral and are covered by the verification above.

---

## 7. Metrics

All clustering metrics use the uncontaminated samples only, from the same
partition the method produced over the full augmented matrix.

| metric | definition | direction |
|---|---|---|
| ARI | agreement with the known partition | higher is better |
| DBI | Davies-Bouldin index of the produced partition | lower is better |
| noise membership | mean largest membership over the injected points | lower is better |
| noise AUC | ROC-AUC of `1 - max_k u_ik` against the injected-point flag, all samples | 0.5 is chance |
| density AUC | ROC-AUC of `-rho` against the injected-point flag; method-independent | 0.5 is chance |
| cluster recall | per-cluster recall after Hungarian matching, min and macro | higher is better |

Dispersion over repetitions is the sample standard deviation (`ddof=1`).
Deterministic methods are reported with zero dispersion.

---

## 8. Citation and contact

If you use this package, please cite the manuscript it accompanies.

Bug reports and questions about the reproduction should reference the
experiment id (`exp01` … `exp11`) and attach the relevant
`results/<experiment>/environment.json`.
