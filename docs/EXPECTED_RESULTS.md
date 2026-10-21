# Expected results

Reference values taken from the saved output of the original experiment
notebook. They are the target for `smoke_test.py` and for a full run of
`exp06`.

---

## 1. Frozen-input comparison protocol

Ten repetitions per benchmark, contaminated matrix generated once and frozen,
initialisation seed varying over 0 … 9.

### ARI mean

| Benchmark | FCM | DW-FCM | LD (Exp) | LD (Rec) | LD (Sig) | MC-FCM |
|---|---|---|---|---|---|---|
| Blobs + Uniform Noise | 0.9347 | 0.9347 | 0.9417 | 0.9347 | 0.9417 | 0.9417 |
| Iris + Outliers | 0.5638 | 0.8015 | 0.8017 | 0.8015 | 0.8340 | 0.7184 |
| Moons + Gaussian Noise | 0.3499 | 0.3659 | 0.3741 | 0.3659 | 0.3907 | 0.3499 |
| Anisotropic + Noise | 0.5179 | 0.5942 | 0.5850 | 0.5997 | 0.6144 | 0.5179 |
| Breast Cancer + Outliers | 0.6704 | 0.6442 | 0.7432 | 0.7313 | 0.6163 | 0.7062 |

`smoke_test.py` checks exactly this block and reports a per-method delta. All
thirty values reproduce to four decimals.

### ARI mean, remaining methods

| Benchmark | AWDFCM | Adaptive FCM-GE | EM-IFCM | k-means++ | GMM | DBSCAN (Opt) | Agglomerative |
|---|---|---|---|---|---|---|---|
| Blobs | 0.9346 | 0.9417 | 0.6062 | 0.9417 | 0.9488 | 0.6914 | 0.9486 |
| Iris | 0.7323 | 0.5638 | 0.5635 | 0.5638 | 0.5681 | 0.6386 | 0.5681 |
| Moons | 0.4337 | 0.3823 | 0.3579 | 0.3823 | 0.0000 | 1.0000 | 0.7159 |
| Anisotropic | 0.5195 | 0.5303 | 0.4627 | 0.5418 | 0.5698 | 0.9901 | 0.4082 |
| Breast | 0.3649 | 0.6236 | 0.0000 | 0.6179 | 0.0673 | 0.3042 | 0.5542 |

EM-IFCM carries a non-zero standard deviation on Blobs (±0.1081) and
Anisotropic (±0.0606); GMM shows a `0.0000` ARI with a `NaN` DBI on Moons,
which is a collapsed partition, not a rounding artefact.

### Noise diagnostics, selected methods

| Benchmark | Method | Noise membership | Noise AUC |
|---|---|---|---|
| Blobs | LD-AMFCM (Exp) | 0.5209 | 0.9536 |
| Blobs | FCM | 0.6640 | 0.9058 |
| Iris | LD-AMFCM (Sig) | 0.3808 | 0.9930 |
| Iris | FCM | 0.6270 | 0.9090 |
| Anisotropic | LD-AMFCM (Sig) | 0.4543 | 0.9057 |
| Breast | LD-AMFCM (Exp) | 0.5276 | 0.9659 |
| Breast | FCM | 0.5922 | 0.9066 |

### Overall summary across the five benchmarks

| Method | ARI mean | ARI std | DBI mean | Noise membership | Noise AUC |
|---|---|---|---|---|---|
| LD-AMFCM (Exp) | 0.6891 | 0.2176 | 0.8083 | 0.5226 | 0.9608 |
| LD-AMFCM (Rec) | 0.6866 | 0.2163 | 0.8124 | 0.5179 | 0.9526 |
| LD-AMFCM (Sig) | 0.6794 | 0.2146 | 0.8260 | 0.4951 | 0.9351 |
| MC-FCM | 0.6468 | 0.2238 | 0.7990 | 0.6128 | 0.9388 |
| DW-FCM | 0.6681 | 0.2158 | 0.8219 | 0.5983 | 0.8837 |
| FCM (Baseline) | 0.6073 | 0.2164 | 0.7445 | 0.6415 | 0.8673 |
| K-Means++ | 0.6095 | 0.2054 | 0.7417 | 1.0000 | 0.5000 |
| Adaptive FCM-GE | 0.6083 | 0.2065 | 0.7423 | 1.0000 | 0.5000 |
| Agglomerative | 0.6390 | 0.2045 | 0.7801 | 1.0000 | 0.5000 |
| AWDFCM | 0.5970 | 0.2339 | 1.0098 | 0.7589 | 0.7701 |
| DBSCAN (Opt) | 0.7248 | 0.2879 | 0.8279 | 0.0000 | 0.8388 |
| GMM | 0.4308 | 0.3950 | 0.8539 | 0.9705 | 0.3898 |
| EM-IFCM | 0.3981 | 0.2424 | 0.5563 | 0.7695 | 0.8895 |

These per-benchmark summary statistics are averaged over the five benchmarks,
so they carry a between-benchmark spread (std ≈ 0.2) that is much larger than
the between-repetition spread. Only the latter is a run-to-run variability
estimate.

---

## 2. Verification status

Verified on the reported environment (Python 3.7.0, NumPy 1.15.1,
scikit-learn 0.19.2). The package was run in full and its tables compared
against the saved notebook output.

| Experiment | What was compared | Result |
|---|---|---|
| `smoke_test.py` | ARI of six methods on all five benchmarks, frozen input | **30 / 30 values match to four decimals** |
| `verify_published_sweeps.py` | default configuration: `exp04` vs `exp05` at the shared default point, plus `exp02` against the notebook | **39 / 39 values match** |
| `verify_published_sweeps.py --legacy` | `exp01` (90), `exp02` (21) and `exp04` (54) against the notebook's saved output, using the negative-power form | **165 / 165 values match** |
| `exp06` | ARI, DBI, noise membership and noise AUC, mean and std, for thirteen methods on five benchmarks | **260 / 260 values match to four decimals** |
| `exp11` | ARI, DBI and noise membership, mean and std, for five methods on five benchmarks | **75 / 75 values match to four decimals**, including the non-zero spreads |

The sweep family (`exp01`, `exp02`, `exp04`) only reproduces with the
negative-power membership form and a single k-means++ restart. The comparison
family (`exp06`, `exp11`) only reproduces with the ratio form and ten
restarts. Both are registered per experiment; see `KNOWN_VARIATIONS.md` §14.

`exp11` also confirms the global-generator faithful mode: the standard
deviations only match if each run consumes the random stream in the same
order as the original, which is the strongest available check that the
reproduction is exact rather than merely close.

To repeat any of them:

```bash
python smoke_test.py
python experiments/exp02_noise_count_stress.py
python experiments/exp06_baseline_comparison.py     # ~5 min on 16 cores
python experiments/exp11_paired_statistics.py
```

and compare the CSVs under `results/<experiment>/` against the tables above.
AWDFCM's re-initialisation source and the reference-tuned DBSCAN
configuration also reproduce, as the `exp06` row shows.

---

## 3. Experiments expected to differ

| Experiment | Why | Action |
|---|---|---|
| `exp01`, `exp04` | the original sweep used the negative-power membership form, `KNOWN_VARIATIONS.md` §14; the default now uses the ratio form | quote the default outputs; 13 of 90 and 6 of 54 grid points change, all on Breast |
| `exp05` | the original sweep used a density estimator that does not match the manuscript, `KNOWN_VARIATIONS.md` §1 | quote the default (paper) output; the legacy table is behind `--legacy-variant` |
| `exp09` | contamination drawn without an explicit seed originally, §5 | quote the regenerated numbers |
| `exp10` | same, plus timings are machine specific | quote relative magnitudes only |

Everything else reproduces the published values. The historical tables for
`exp01`, `exp02`, `exp04` and `exp05` remain reproducible with
`--legacy-membership` / `--legacy-variant`.

### `exp05` default point after the density alignment

On Iris with the exponential mapping, `k = 12`, `m_min = 1.05`,
`m_max = 3.5`:

| configuration | ARI |
|---|---|
| `exp04` (k sweep, `k = 12`) | 0.8176 |
| `exp05` default, `m_min = 1.05` | **0.8176** |
| `exp05` default, `m_max = 3.5` | **0.8176** |
| `exp05 --legacy-variant` | 0.5638 |

The three tables now agree at the shared default point.

---

## 4. Timings

`exp10` on the contaminated Breast benchmark (n = 599, p = 30, C = 2):

| Method | Density estimation | Iteration | Total |
|---|---|---|---|
| FCM | 0.0000 | 0.0200 | 0.0200 |
| DW-FCM | 0.0203 | 0.0298 | 0.0500 |
| LD-AMFCM (Exp) | 0.0203 | 0.0301 | 0.0503 |
| LD-AMFCM (Rec) | 0.0203 | 0.0199 | 0.0402 |
| LD-AMFCM (Sig) | 0.0203 | 0.0400 | 0.0603 |

The intended reading is the ratio: the density-derived quantities are a
one-off cost of the same order as the iteration itself, after which LD-AMFCM
costs the same per iteration as the other methods.

These iteration times are about eight times smaller than the originally
reported values because the membership update is vectorised; the memberships
are bit-identical between the two forms. See `KNOWN_VARIATIONS.md` §7.

Each entry is a single measurement in the 20–40 ms range, so run-to-run
variation is a meaningful fraction of the difference between any two rows.
Treat ordering within the table as indicative rather than established, and
average several repetitions before making a timing claim.

`exp03` per-sample-size results with a sequential kNN search
(`--n-jobs 1`); the default `n_jobs=-1` path measures worker-pool start-up
instead and is discussed in `KNOWN_VARIATIONS.md` §13:

| n | density (s) | LD-AMFCM total (s) | FCM total (s) | Iterations (Exp / Rec) |
|---|---|---|---|---|
| 500 | 0.005 | 0.013 | 0.008 | 10.8 / 72 |
| 1 000 | 0.020 | 0.031 | 0.014 | 8.0 / 58 |
| 2 000 | 0.077 | 0.090 | 0.019 | 7.0 / 62 |
| 5 000 | 0.502 | 0.525 | 0.040 | 6.0 / 55 |
| 10 000 | 1.670 | 1.714 | 0.072 | 5.0 / 61 |

The reciprocal mapping needs roughly ten times as many iterations as the
exponential and sigmoid mappings at every sample size. This is a property of
the mapping, not of the sample size, and is worth stating separately if the
paper discusses convergence.
