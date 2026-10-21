"""
Check the parameter-sweep tables.

Two modes:

**default** -- checks that the package is internally consistent under the
manuscript configuration.  The ``k`` sweep (``exp04``) and the two fuzzifier
sweeps (``exp05``) share a default point; all three must agree there.  The
noise-count stress test (``exp02``) is unchanged by the membership-form switch
and is checked against the original notebook's saved output.

**``--legacy``** -- checks the historical reproduction path.  Runs
``exp01``, ``exp02`` and ``exp04`` with ``--legacy-membership`` (the
negative-power form used by the original notebook) and compares all 165
values against the notebook's saved output.  This is the path that reproduces
the original curves; the default path deliberately does not, because it uses
the numerically stable form that matches the manuscript and the headline
comparison.

Usage
-----
    python verify_published_sweeps.py            # consistency check
    python verify_published_sweeps.py --legacy   # reproduce the old curves
    python verify_published_sweeps.py --run      # run experiments first
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS = PACKAGE_ROOT / "results"

TOL = 1e-4


# ----------------------------------------------------------------------
# Reference values from the notebook's saved output (legacy path only)
# ----------------------------------------------------------------------

EXP01 = {
    "Breast Cancer + Outliers": {
        "alpha": [0.6274, 0.7313, 0.7311, 0.7676, 0.7612, 0.7672, 0.7547,
                  0.7182, 0.6941, 0.7058],
        "beta": [0.5312, 0.6329, 0.7135, 0.7313, 0.7373, 0.7251, 0.7311,
                 0.7311, 0.7251, 0.7371],
        "gamma": [0.5467, 0.6054, 0.6053, 0.6108, 0.6386, 0.7019, 0.7017,
                  0.7437, 0.7868, 0.7866],
    },
    "Moons + Gaussian Noise": {
        "alpha": [0.4162, 0.4162, 0.3991, 0.3907, 0.3907, 0.3907, 0.3907,
                  0.3907, 0.3907, 0.3907],
        "beta": [0.4337, 0.4162, 0.4162, 0.3991, 0.3991, 0.3991, 0.3991,
                 0.3991, 0.3991, 0.3991],
        "gamma": [0.4337, 0.4426, 0.4337, 0.4337, 0.4337, 0.4337, 0.4337,
                  0.4337, 0.4337, 0.4249],
    },
    "Iris + Outliers": {
        "alpha": [0.8015, 0.8015, 0.8176, 0.8176, 0.7163, 0.7163, 0.7163,
                  0.7163, 0.7163, 0.7163],
        "beta": [0.8015] * 10,
        "gamma": [0.8015, 0.8015, 0.8015, 0.8015, 0.8176, 0.8176, 0.8176,
                  0.8176, 0.8176, 0.8176],
    },
}

EXP02 = {
    40: {"FCM": 0.677689, "FCM + m_i": 0.703824, "LD-AMFCM": 0.630203},
    60: {"FCM": 0.627006, "FCM + m_i": 0.701284, "LD-AMFCM": 0.642523},
    80: {"FCM": 0.606662, "FCM + m_i": 0.701721, "LD-AMFCM": 0.668130},
    100: {"FCM": 0.543048, "FCM + m_i": 0.702667, "LD-AMFCM": 0.688867},
    120: {"FCM": 0.493283, "FCM + m_i": 0.705960, "LD-AMFCM": 0.695872},
    140: {"FCM": 0.448711, "FCM + m_i": 0.679056, "LD-AMFCM": 0.702909},
    160: {"FCM": 0.381208, "FCM + m_i": 0.680059, "LD-AMFCM": 0.707619},
}

EXP04 = {
    "Breast": {5: (.7614, .7374, .6442), 8: (.7614, .7313, .6163),
               12: (.7676, .7313, .6108), 16: (.7676, .7313, .6053),
               20: (.7676, .7313, .6053), 30: (.7676, .7313, .6054)},
    "Iris": {5: (.7862, .8176, .8176), 8: (.8176, .8015, .8015),
             12: (.8176, .8015, .8015), 16: (.8176, .8015, .8176),
             20: (.8176, .8015, .8176), 30: (.8176, .8015, .8176)},
    "Moons": {5: (.3991, .4076, .4426), 8: (.3991, .4076, .4426),
              12: (.3991, .3991, .4337), 16: (.3991, .4076, .4337),
              20: (.4076, .4076, .4337), 30: (.4162, .4162, .4337)},
}


class Checker:
    def __init__(self):
        self.total = 0
        self.failed = 0

    def check(self, name, got, want):
        self.total += 1
        if abs(got - want) >= TOL:
            self.failed += 1
            print("  MISMATCH {}: got {:.5f}, want {:.5f}".format(
                name, got, want))


# ----------------------------------------------------------------------
# Legacy path: reproduce the notebook
# ----------------------------------------------------------------------

def check_legacy(checker):
    df = pd.read_csv(RESULTS / "exp01_mapping_sensitivity" /
                     "mapping_sensitivity_legacy.csv")
    for dataset, params in EXP01.items():
        for param, refs in params.items():
            part = (df[(df.Dataset == dataset) & (df.Parameter == param)]
                    .sort_values("Value"))
            for value, got, want in zip(part.Value, part.ARI_Mean, refs):
                checker.check(
                    "exp01 {}/{}={:.2f}".format(dataset, param, value),
                    got, want)

    df = pd.read_csv(RESULTS / "exp02_noise_count_stress" /
                     "noise_count_stress_summary_legacy.csv")
    for level, methods in EXP02.items():
        row = df[df.Noise_Count == level].set_index("Method")["ARI_Mean"]
        for method, want in methods.items():
            checker.check("exp02 n={}/{}".format(level, method),
                          row[method], want)

    df = pd.read_csv(RESULTS / "exp04_k_sensitivity" /
                     "k_sensitivity_single_run_legacy.csv")
    for dataset, by_k in EXP04.items():
        for k, refs in by_k.items():
            for mapping, want in zip(("Exp", "Rec", "Sig"), refs):
                got = float(df[(df.Dataset == dataset) & (df.k == k) &
                               (df.Mapping == mapping)].ARI)
                checker.check(
                    "exp04 {}/k={}/{}".format(dataset, k, mapping), got, want)


# ----------------------------------------------------------------------
# Default path: internal consistency under the manuscript configuration
# ----------------------------------------------------------------------

def check_consistency(checker):
    k = pd.read_csv(RESULTS / "exp04_k_sensitivity" /
                    "k_sensitivity_single_run.csv")
    f = pd.read_csv(RESULTS / "exp05_fuzzifier_sensitivity" /
                    "fuzzifier_sensitivity_paper.csv")

    for dataset in ("Breast", "Iris", "Moons"):
        for mapping in ("Exp", "Rec", "Sig"):
            k_value = float(k[(k.Dataset == dataset) & (k.k == 12) &
                              (k.Mapping == mapping)].ARI)

            for parameter, value in (("m_min", 1.05), ("m_max", 3.5)):
                part = f[(f.Dataset == dataset) & (f.Mapping == mapping) &
                         (f.Parameter == parameter) & (f.Value == value)]
                f_value = float(part.ARI)
                checker.check(
                    "default point {}/{} vs {}".format(
                        dataset, mapping, parameter),
                    f_value, k_value)

    # The noise-count stress test is unchanged by the membership switch.
    df = pd.read_csv(RESULTS / "exp02_noise_count_stress" /
                     "noise_count_stress_summary.csv")
    for level, methods in EXP02.items():
        row = df[df.Noise_Count == level].set_index("Method")["ARI_Mean"]
        for method, want in methods.items():
            checker.check("exp02 n={}/{}".format(level, method),
                          row[method], want)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy", action="store_true",
                        help="verify the negative-power reproduction path "
                             "against the notebook's saved output")
    parser.add_argument("--run", action="store_true",
                        help="run exp01, exp02 and exp04 first (with "
                             "--legacy-membership when --legacy is given)")
    args = parser.parse_args()

    if args.run:
        flag = ["--legacy-membership"] if args.legacy else []
        for script in ("exp01_mapping_sensitivity.py",
                       "exp02_noise_count_stress.py",
                       "exp04_k_sensitivity.py"):
            print("running {} {}...".format(script, " ".join(flag)))
            completed = subprocess.run(
                [sys.executable, str(PACKAGE_ROOT / "experiments" / script)]
                + flag,
                cwd=str(PACKAGE_ROOT),
            )
            if completed.returncode != 0:
                print("  failed with exit code {}".format(completed.returncode))
                return 1

    checker = Checker()
    if args.legacy:
        check_legacy(checker)
        label = "legacy reproduction vs notebook"
    else:
        check_consistency(checker)
        label = "default configuration consistency"

    print("{}: {} of {} values match, {} mismatches".format(
        label, checker.total - checker.failed, checker.total,
        checker.failed))
    return 0 if checker.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
