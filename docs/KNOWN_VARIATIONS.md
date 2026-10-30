# Known variations, preserved configuration and repairs

A reproducibility package has to answer two questions at once: can a reader
regenerate the published numbers, and is the released code correct? Those pull
in opposite directions when the original experiments contain inconsistencies.
This file states, item by item, which variants were **preserved** because the
published numbers depend on them, and which were **repaired** because the
original behaviour was an accident of execution rather than a design choice.

Every repair is algebraically or statistically neutral, and all of them are
covered by `smoke_test.py`, which reproduces the published ARI values.

---

## 1. Resolved: the `m_min` / `m_max` sweep now follows the manuscript

**Affects:** `exp05`.

The original notebook produced these two tables with an implementation that
does not match the density estimator defined in the manuscript:

| | manuscript / headline configuration | original sweep implementation |
|---|---|---|
| density estimator | σ = max{global median of kNN distances, 1e-12}, ρ = **mean** of Gaussian kernels | σ_i = mean distance to the k neighbours (per point), ρ = **sum** of Gaussian kernels |
| initialisation | k-means++ prototypes | random normalised memberships, `RandomState(42)`, prototypes start at zero |
| stopping rule | ‖V_new − V‖_F < 1e-4 | ‖U_new − U‖_F < 1e-5 |
| distances in the membership step | squared, stabilised | Euclidean, stabilised |

**Why it matters.** The variant's normalised density is concentrated near one
(median 0.894 against 0.517 on Iris). Under the exponential mapping with
α = 6 that pins 96.5 % of the samples at m_i ≈ m_min = 1.05, so the
density-derived weights carry no information — they give the same ARI as
uniform weights — and the whole m_max sweep collapses to a single value. At
the nominal default point the two implementations disagree:

| Iris, exponential mapping, k = 12, m_min = 1.05, m_max = 3.5 | ARI |
|---|---|
| manuscript density, either solver | 0.8176 |
| original sweep density, either solver | 0.5638 |

Isolation shows the density estimator alone is responsible: the initialisation
and the membership form make no difference at this point.

**What was done.** `exp05` now defaults to the manuscript estimator and the
headline solver, so a change in `m_min` or `m_max` is the only difference from
the default point. Its output is `fuzzifier_sensitivity_paper.csv` and the
default point now equals the `k` sweep's 0.8176. The original implementation
is retained behind `--legacy-variant`, which writes
`fuzzifier_sensitivity_legacy_variant.csv`, so the old notebook tables remain
reproducible.

**What this means for the paper.** The m_min and m_max tables must be
regenerated from the default run, not quoted from the original notebook. The
old m_max table in particular showed an apparent insensitivity that was an
artefact of the estimator, not a property of the method.

---

## 2. Preserved: DBSCAN is a reference-tuned upper bound

**Affects:** `exp06`.

DBSCAN's `eps` and `min_samples` are searched over a declared grid
(20 × 9 = 180 configurations) and the configuration with the best ARI on the
clean ground truth is reported. Every other method in the table uses a
configuration fixed in advance.

**What was done.** The behaviour is preserved, the grid is declared in the
parameter register, and the method is labelled `DBSCAN (Opt)`.

**What this means for the paper.** The row is an oracle-tuned reference point,
not an untuned competitor, and the table legend should say so. The label
already carries `(Opt)`; a footnote is recommended.

---

## 3. Preserved: the two benchmark-generator seeding conventions

**Affects:** `exp01`, `exp04`, `exp05` versus `exp06`, `exp08`, `exp11`.

Two conventions appear and they produce **different contaminating points for
the same benchmark name**:

* `local_rng=True` — each benchmark draws from a freshly seeded generator
  `RandomState(42)`. Used by the single-benchmark cells, where one benchmark
  was generated per executed cell.
* `local_rng=False` — all benchmarks draw from one stream seeded once, in the
  declared benchmark order. Used by the multi-benchmark studies, where the
  later benchmarks consume a stream already advanced by the earlier ones.

**What was done.** Both are implemented and each experiment uses the
convention it was run with. `datasets.get_benchmark(..., local_rng=False)`
carries an explicit warning in its docstring, because using it to fetch a
single benchmark silently yields the *first-in-order* contamination rather
than the one used in the comparison study.

---

## 4. Preserved: the global-generator dependence in the paired study

**Affects:** `exp11`.

The original repeated-run study seeded the global NumPy generator once per
repetition and then let the k-means++ initialisation draw from that same
stream, by calling `KMeans(n_init=10)` without `random_state`. A method's
initial prototypes therefore depend on how much randomness the earlier steps
of that run consumed, which makes the run reproducible only as a whole, under
the documented execution order.

**What was done.** The behaviour is preserved as the default, because the
published statistics depend on it, and the fact is stated in the docstring and
the seed register. `--explicit-init-seed` passes the run index as an explicit
initialisation seed, which removes the coupling and is the form to use if the
study is ever re-run. The two modes give different numbers.

---

## 5. Repaired: contamination drawn without resetting the generator

**Affects:** `exp09`, `exp10`.

In the original notebook these two cells injected contamination without
resetting the generator first, so the injected points depended on the
interpreter's global random state at the moment the cell was executed — a
state that is not recoverable from the notebook.

**What was done.** Both scripts now call `np.random.seed(42)` before the
sweep, which makes them reproducible. The original state cannot be
reconstructed, so the numbers in these two tables may move slightly relative
to whatever was committed to the paper. Both tables should be regenerated from
this package before the numbers are quoted.

This is the only place in the package where a change may alter published
output rather than merely reorganise it.

---

## 6. Repaired: a convergence test that compared a quantity with itself

**Affects:** the random-membership solver, used by `exp05 --legacy-variant`
only. The default `exp05` configuration does not use this solver.

The original membership-update loop assigned the new memberships to the
variable before testing the change, so the test evaluated
`‖U_new − U_new‖ < tol`, which is unconditionally true and terminated the loop
after exactly one iteration.

**What was done.** The change is computed before the assignment, so the
solvers iterate until the memberships actually stabilise. This is the
documented intent of the variant's `tol = 1e-5` setting; under the original
behaviour that setting had no effect and the parameter was inert.

**Consequence.** The legacy table regenerated here differs from the originally
reported sweep, because the original code did not do what its own parameter
said it did. This repair no longer affects any default experiment: item 1
moved `exp05` onto the manuscript estimator and the headline solver, neither of
which uses this code path. It matters only when reproducing the historical
table with `--legacy-variant`.

---

## 7. Repaired: the membership update divided a row by itself

**Affects:** the headline solver, all experiments.

The membership update was originally written as a per-sample Python loop
evaluating `dist[i, j] / dist[i]`, a scalar over an array. Re-expressing the
same loop as a vectorised expression invites writing `dist[i, :] / dist[i]`,
which divides the row by itself and returns ones for every entry, giving
uniform memberships and an ARI of exactly zero for every method.

**What was done.** The update is implemented as a pairwise distance ratio,
`ratios[i, j, k] = d_ij / d_ik`, with the exponent summed over `k`. This is
the same arithmetic as the loop, with the same result — confirmed by
`smoke_test.py` — and it is roughly two orders of magnitude faster for the
large sample sizes in `exp03`.

**Measured impact on the reported timings.** On the `exp10` setup (Breast
Cancer plus 30 outliers, n = 599, p = 30, C = 2, ten k-means++ restarts):

| solver | iteration time |
|---|---|
| original per-sample loop | 0.1688 s |
| vectorised (released package) | 0.0216 s |

The two produce **bit-identical membership matrices**
(`max |U_loop − U_vectorised| = 0.0`) and identical hard labels. The
iteration-time column of the running-time table therefore drops by roughly a
factor of eight relative to the originally reported values, while the method
itself is unchanged. Any timing claim that compares the two revisions should
say so; the published table's `Iteration Time` column reports the loop
implementation, and the numbers in this package report the vectorised one.

A side effect worth noting: with the loop implementation the preprocessing
step looked negligible next to the iteration cost (0.015 s against
0.186–0.356 s), whereas with the vectorised implementation the two are
comparable (0.020 s against 0.020–0.040 s). The preprocessing is a one-off
cost of the same order as the iteration, which is a different and stronger
statement than "negligible".

---

## 8. Noted: three different stopping rules across experiments

| Rule | Where |
|---|---|
| ‖V_new − V‖_F < 1e-4 | all main experiments |
| max_k ‖v_k^new − v_k‖ < 1e-4 | `exp03` (scalability) |
| ‖U_new − U‖_F < 1e-5 | `exp05 --legacy-variant` only |

All three use the same tolerance of 1e-4 except the variant. The scalability
study's rule is strictly stricter than the Frobenius rule for the same
tolerance, since it bounds the largest per-cluster shift.

**What was done.** Each experiment uses the rule it was run with, and the rule
is stated in the parameter register. Convergence behaviour on a well-separated
problem usually reaches the same fixed point under all three, which is why
this is noted rather than repaired.

---

## 9. Resolved: `n_init` unified to ten, except the timing study

The original notebook used ten k-means++ restarts in the comparison, ablation,
density-reliability, noise-ratio, runtime and paired-statistics experiments,
but one restart in the mapping, `k`, noise-count and scalability experiments.
The split had no methodological justification — it was an artefact of how the
code evolved — and made the same estimator look differently configured across
tables.

**What was done.** Every result experiment now uses ten restarts, matching the
headline comparison. The scalability study keeps one restart because its
timing columns measure the cost of the solver itself and a ten-fold
initialisation budget would distort them. The `--legacy-membership`
reproduction paths pin the count to one, matching the notebook's protocol.

**Evidence that this changes nothing.** One and ten restarts were compared
across the complete grids, under both membership forms:

| Path | Comparisons | Differences |
|---|---|---|
| `exp01` default (ratio) | 90 grid points | 0 |
| `exp02` default (ratio) | 210 per-run values | 0 |
| `exp04` default (ratio) | 54 grid points | 0 |
| `exp05` default (ratio) | 108 grid points | 0 |
| `exp01` legacy (negative-power) | 90 grid points | 0 |
| `exp02` legacy (negative-power) | 210 per-run values | 0 |
| `exp04` legacy (negative-power) | 54 grid points | 0 |

All 816 comparisons are bit-identical. The unification removes a
configuration difference without changing any reported number.

---

## 10. Noted: the degenerate Davies-Bouldin value

The Davies-Bouldin index returns `0` when fewer than two clusters are present.
The comparison table converts that case to `NaN` before aggregating, so a
degenerate partition is visibly excluded rather than silently scored as a
perfect `0`. The density-reliability stress test uses the raw form, matching
how it was originally reported.

**What was done.** Both access paths exist (`davies_bouldin_index` and
`safe_davies_bouldin_index`) and each experiment uses the one it was run with.
This matters for interpretation: in the comparison table a `NaN` DBI means the
method collapsed, whereas in the stress table a `0` may mean the same thing.

---

## 11. Noted: the degenerate normalisation constant

The min-max normalisation of the density adds a 1e-12 guard, so a fully
degenerate density collapses to zero rather than raising. One original cell
used a separate guard returning a constant 0.5 instead.

**What was done.** The single form is used everywhere. The difference is
unreachable for the reported benchmarks, since a degenerate density requires
every sample to have identical distance structure.

---

## 12. Noted: the AWDFCM centre re-initialisation source

When two AWDFCM centres coincide below 1e-3, the replacement point is drawn
from a freshly constructed `RandomState(random_state)` in one version of the
code and from the global generator in another. The former is deterministic and
is what the reported ten-run table used.

**What was done.** `awdfcm(..., reinit_mode="fresh_rng")` is the default and
reproduces the reported table; `reinit_mode="global"` restores the other
version. Preserved rather than repaired because the published numbers depend
on it, and noted because a fresh generator per re-initialisation always
returns the same replacement point, which is unlikely to be what was intended.

---

## 13. Noted: the scalability timings measure thread start-up, not the estimator

**Affects:** `exp03`.

The scalability study ran its exact kNN search with `n_jobs=-1`. In this
environment scikit-learn 0.19.2 dispatches that through a joblib worker pool
whose start-up cost swamps the computation:

| n | `n_jobs=-1` | `n_jobs=1` | ratio |
|---|---|---|---|
| 500 | 3.95 s | 0.005 s | ~760× |
| 10 000 | 5.54 s | 1.72 s | ~3× |

The consequence for the reported figure is not cosmetic. Density estimation
appears to dominate the runtime at every sample size — 2.2 s out of 2.2 s
total at n = 500 — when the actual computation at that size is five
milliseconds. A reader would conclude that the density step is the bottleneck
of the method; on this evidence it is not.

**What was done.** The default is left at `n_jobs=-1` so the published timing
columns remain reproducible, `--n-jobs 1` produces timings that reflect the
algorithm, and the caveat is written into the experiment's
`environment.json`. `results/exp03_scalability/` holds the sequential run.

**What the corrected numbers show.** With a sequential kNN search the density
step costs 0.005 s at n = 500 and 1.67 s at n = 10 000, against 0.072 s total
for plain FCM at the largest size:

| n | density | LD-AMFCM total (Exp) | FCM total | density share |
|---|---|---|---|---|
| 500 | 0.005 s | 0.013 s | 0.008 s | 41 % |
| 1 000 | 0.020 s | 0.031 s | 0.014 s | 65 % |
| 2 000 | 0.077 s | 0.090 s | 0.019 s | 86 % |
| 5 000 | 0.502 s | 0.525 s | 0.040 s | 96 % |
| 10 000 | 1.670 s | 1.714 s | 0.072 s | 97 % |

So the qualitative conclusion survives: the density step genuinely dominates
at large n and grows super-linearly, because the exact brute-force kNN search
is quadratic in n. What does **not** survive is the small-n end of the
published figure, where the density step appeared to dominate when it in fact
cost five milliseconds. The crossover to genuine dominance is around n ≈ 500,
not at n = 500 already being 99.5 %.

**What this means for the paper.** Regenerate the figure with
`python experiments/exp03_scalability.py --n-jobs 1` before the timing claim
goes to print. The framing that is defensible is that the density step is a
one-off O(n²) cost that dominates as n grows, which also motivates an
approximate kNN index over the exact search.

**Unrelated observation from the same run.** The reciprocal mapping needs
roughly ten times as many iterations as the other two — about 60 against 5 to
11 at every sample size — while reaching a comparable objective. Iteration
counts are recorded per run, so this is visible in the raw output. If the
paper argues anything about convergence speed, it should be stated per
mapping rather than for LD-AMFCM as a whole.

---

## 14. Two membership-update forms, and the 1e-12 guard between them

**Affects:** every experiment, but differently.

The notebook contains two algebraically equivalent membership updates, and
they do **not** produce the same numbers in general:

**Ratio form** (cells 8–15: baseline comparison, ablation, stress, noise
ratio, runtime, paired study)

```
u_ij = 1 / sum_k (d_ij / d_ik) ** (1 / (m_i - 1))
```

Every row sums to exactly one by construction.

**Negative-power form** (cells 0–7: mapping sweep, k sweep, noise-count
stress, scalability, fuzzifier sweeps)

```
inv = d ** (-1 / (m_i - 1));   u = inv / (sum(inv) + 1e-12)
```

The two agree whenever the powers stay above ~1e-12. They diverge for a
*crisp* sample: as `m_i -> 1.05` the exponent `1/(m_i - 1)` reaches 20, and
for a sample whose distances are large the negative powers fall far below
1e-12. At that point the additive guard is no longer a rounding protector but
the dominant term of the denominator, so the row stops summing to one. On the
Breast benchmark with `alpha = 6.33`, 157 of 599 rows are affected and their
row sums fall to ~1e-6, meaning those samples contribute essentially nothing
to the prototype update.

**Consequence.** The two forms give different clusterings for exactly the
regimes where the fuzzifier is small, which is where the mapping and `k`
sweeps spend most of their grid:

| experiment | parameter value | ratio form | negative-power form | notebook |
|---|---|---|---|---|
| `exp01` | `alpha = 6.33` | 0.7431 | 0.7676 | 0.7676 |
| `exp01` | `gamma = 30.00` | 0.6442 | 0.7866 | 0.7866 |
| `exp04` | Breast, `k = 12`, Exp | 0.7371 | 0.7676 | 0.7676 |

**What was done.** Both forms are implemented,
`fcm_weighted(..., membership_form=...)`. **Every default experiment now uses
the ratio form**, so the sensitivity analyses and the headline comparison share
one numerical implementation and all tables agree at the shared default point.
The negative-power form is retained behind `--legacy-membership` on `exp01`,
`exp02`, `exp03` and `exp04`, and behind `--legacy-variant` on `exp05`, solely
to regenerate the original notebook's tables. Those historical files — both
the CSVs and the figures — are written with a `_legacy` suffix and are never
mixed with the default results. (An earlier release wrote both configurations
to the same figure filenames, so the legacy run silently overwrote the default
plots; the suffix prevents that.)

Verification, both runnable:

| command | checks | result |
|---|---|---|
| `python verify_published_sweeps.py` | default configuration agrees across `exp04` and `exp05`, and `exp02` against the notebook | **39 / 39** |
| `python verify_published_sweeps.py --legacy` | the negative-power reproduction path against the notebook | **165 / 165** |

**This was a real defect in the first release of this package**, where the
ratio form was used everywhere and the mapping and `k` sweeps therefore did
not reproduce their published curves. It was found only because `exp01`
diverged from the notebook after an unrelated change, which prompted a
value-by-value comparison.

**What this means for the paper.** Two of the three mapping sweeps and the `k`
sweep in the original notebook were produced with a numerically fragile
implementation whose additive guard silently deactivates a subset of the
samples. The default results in this package are now computed with the ratio
form, so they differ from the notebook at 13 of the 90 mapping-grid points and
6 of the 54 k-grid points — all of them on Breast, concentrated at large
`alpha` and large `gamma`. In particular, the Breast exponential `k` curve is
flat at 0.7371 under the ratio form; the variation in the original curve was
an artefact of the guard, not a property of the method. The sensitivity tables
in the manuscript must therefore be regenerated from the default outputs.

**Related observation.** The negative-power form also converges more slowly in
some configurations, plausibly because the same guard suppresses part of the
prototype update. In `exp03` at `p = 20` the exponential mapping needs 58–80
iterations across the five seeds, against 5–13 at every other dimensionality.
The ratio form does not show this.

---

## 15. Summary

| # | Item | Status | May change published numbers |
|---|---|---|---|
| 1 | fuzzifier sweep density estimator | resolved: default follows the manuscript, legacy behind a flag | yes, `exp05` must be regenerated |
| 2 | DBSCAN oracle-tuned | preserved, declared | no |
| 3 | two benchmark seeding conventions | preserved, both implemented | no |
| 4 | paired study depends on the global generator | preserved, flagged | no |
| 5 | contamination drawn without resetting the generator | **repaired** | **yes, `exp09` and `exp10`** |
| 6 | convergence test compared a quantity with itself | **repaired** | only `exp05 --legacy-variant` |
| 7 | membership update divided a row by itself | repaired (vectorisation) | no |
| 8 | three stopping rules | noted | no |
| 9 | `n_init` differs between families | resolved: 10 everywhere except the scalability timing study | no (816 comparisons identical) |
| 10 | degenerate DBI value | noted, both paths available | no |
| 11 | degenerate normalisation constant | unified | no |
| 12 | AWDFCM re-initialisation source | preserved, both available | no |
| 13 | scalability timings measure thread start-up | preserved, `--n-jobs 1` available | no (but the figure misleads) |
| 14 | two membership forms either side of a 1e-12 guard | resolved: default is the ratio form everywhere, legacy behind a flag | yes, the sweeps must be regenerated |
