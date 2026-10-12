# Seed register

Two randomness sources are controlled separately throughout:

* a **data seed** fixes the augmented benchmark matrix — the base samples plus
  the injected contamination;
* an **initialisation seed** fixes the initial prototypes of a method.

A matching integer in both roles does not imply matching inputs or matching
initial prototypes, so the two are declared and reported independently.

The authoritative definition is `ldamfcm/config.py`; the table below is a
rendering of it.

---

## Register

| Experiment | Data seed | Initialisation seed | Runs | Data regenerated per repetition | Protocol |
|---|---|---|---|---|---|
| **exp01** mapping sensitivity, Breast | 42 | 0 … 9 | 10 | no | frozen input |
| **exp01** mapping sensitivity, Moons | 42 | 42 | 1 | no | frozen input |
| **exp01** mapping sensitivity, Iris | 42 | 42 | 1 | no | frozen input |
| **exp02** noise-count stress | 100 + s | s | 10 | yes | regenerated input |
| **exp03** scalability | 42 (generator) | 0 … 4 | 5 | no | frozen synthetic data |
| **exp04** neighbourhood size k | 42 | 42 | 1 | no | frozen input |
| **exp05** fuzzifier bounds | 42 | 42 | 1 | no | frozen input |
| **exp06** baseline comparison | 42 | 0 … 9 | 10 | no | frozen input |
| **exp07** density-reliability stress | 42 + s | 42 + s | 10 | yes | regenerated input |
| **exp08** ablation | 42 | 42 | 1 | no | frozen input |
| **exp09** contamination ratio | 42 | 42 | 1 | yes (per grid point) | regenerated input |
| **exp10** runtime | 42 | 42 | 1 | n/a | single measurement |
| **exp11** paired statistics | s, s = 0 … 9 | s, s = 0 … 9 | 10 | yes | regenerated input |

---

## Notes on individual rows

**exp02 — the two streams are deliberately offset.** The data seed runs from
100 and the initialisation seed from 0, so a given repetition never uses the
same integer for both purposes. If they were equal, a difference between two
repetitions could not be attributed to either source, and the experiment would
be unable to distinguish "different contamination" from "different starting
point".

**exp07 — the streams are coupled.** Both carry the index `42 + s`. This is a
property of the original experiment and is disclosed rather than repaired, so
that the published numbers remain reproducible. `exp02` is the offset
counterpart and the two together cover both designs.

**exp03 — the data seed is constant on purpose.** The synthetic matrices are
held fixed across the repetitions so that the timing differences reflect the
solver rather than the data draw.

**exp06 — matched initialisation.** Within one run index, FCM, DW-FCM, the
three LD-AMFCM mappings and MC-FCM are initialised with the same
`random_state` on the same frozen matrix, so they start from identical
prototypes. AWDFCM, Adaptive FCM-GE and EM-IFCM follow their own published
initialisation rules and start elsewhere by design.

**exp11 — the only place where an initialisation seed is not an argument.**
The original study seeded the global NumPy generator once per repetition and
then let the k-means++ initialisation draw from that same global stream, so a
method's starting point depends on how much randomness the earlier steps of
that run consumed. The package reproduces this exactly by passing
`random_state=None`. Run with `--explicit-init-seed` to replace it with the
clean form, in which the run index is passed as the initialisation seed. The
two modes give different numbers; the default mode is the one that reproduces
the published statistics.

---

## Diagnostic: is a run actually reproducible?

Two failure modes this register is designed to make visible:

1. **A seed that is set but never used for the thing it appears to control.**
   Reusing the integer 42 for both roles looks tidy and proves nothing.
2. **A draw that inherits global state.** If a data draw is not preceded by an
   explicit seeding, its result depends on what the interpreter did earlier.
   `exp09` and `exp10` inherited an unseeded draw in the original notebook;
   both now reset the generator explicitly, and the change is recorded in
   `KNOWN_VARIATIONS.md`.

The `environment.json` written by every experiment records the seeds actually
used, so a mismatch can be diagnosed from the artefacts rather than by
re-reading the source.
