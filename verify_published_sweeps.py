"""
Check the parameter-sweep tables against the values saved in the original
experiment notebook.

``smoke_test.py`` re-runs the baseline comparison and checks it live.  This
script instead reads the CSVs that ``exp01``, ``exp02`` and ``exp04`` wrote
and compares them, value by value, against the numbers recorded in the
notebook's saved output.

Run the three experiments first, or with ``--run`` to do both:

    python verify_published_sweeps.py
    python verify_published_sweeps.py --run

These sweeps are the family that needs the negative-power membership form and
a single k-means++ restart to reproduce; see docs/KNOWN_VARIATIONS.md.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS = PACKAGE_ROOT / "results"

TOL = 5e-5


# ----------------------------------------------------------------------
# Reference values from the notebook's saved output
# ----------------------------------------------------------------------

# exp01: alpha / beta / gamma grids, ten repetitions on Breast, single run on
# Moons and Iris.
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

# exp02: ARI mean at seven contamination levels, three methods.
EXP02 = {
    40: {"FCM": 0.677689, "FCM + m_i": 0.703824, "LD-AMFCM": 0.630203},
    60: {"FCM": 0.627006, "FCM + m_i": 0.701284, "LD-AMFCM": 0.642523},
    80: {"FCM": 0.606662, "FCM + m_i": 0.701721, "LD-AMFCM": 0.668130},
    100: {"FCM": 0.543048, "FCM + m_i": 0.702667, "LD-AMFCM": 0.688867},
    120: {"FCM": 0.493283, "FCM + m_i": 0.705960, "LD-AMFCM": 0.695872},
    140: {"FCM": 0.448711, "FCM + m_i": 0.679056, "LD-AMFCM": 0.702909},
    160: {"FCM": 0.381208, "FCM + m_i": 0.680059, "LD-AMFCM": 0.707619},
}

# exp04: ARI at six neighbourhood sizes, three mappings, three datasets.
# Order within each tuple is Exp, Rec, Sig.
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
            print("  MISMATCH {}: got {:.4f}, want {:.4f}".format(
                name, got, want))


def check_exp01(checker):
    path = RESULTS / "exp01_mapping_sensitivity" / "mapping_sensitivity.csv"
    df = pd.read_csv(path)
    for dataset, params in EXP01.items():
        for param, refs in params.items():
            part = (df[(df.Dataset == dataset) & (df.Parameter == param)]
                    .sort_values("Value"))
            for value, got, want in zip(part.Value, part.ARI_Mean, refs):
                checker.check(
                    "exp01 {}/{}={:.2f}".format(dataset, param, value),
                    got, want)


def check_exp02(checker):
    path = RESULTS / "exp02_noise_count_stress" / "noise_count_stress_summary.csv"
    df = pd.read_csv(path)
    for level, methods in EXP02.items():
        row = df[df.Noise_Count == level].set_index("Method")["ARI_Mean"]
        for method, want in methods.items():
            checker.check("exp02 n={}/{}".format(level, method),
                          row[method], want)


def check_exp04(checker):
    path = RESULTS / "exp04_k_sensitivity" / "k_sensitivity_single_run.csv"
    df = pd.read_csv(path)
    for dataset, by_k in EXP04.items():
        for k, refs in by_k.items():
            for mapping, want in zip(("Exp", "Rec", "Sig"), refs):
                got = float(df[(df.Dataset == dataset) & (df.k == k) &
                               (df.Mapping == mapping)].ARI)
                checker.check(
                    "exp04 {}/k={}/{}".format(dataset, k, mapping), got, want)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true",
                        help="run exp01, exp02 and exp04 first")
    args = parser.parse_args()

    if args.run:
        for script in ("exp01_mapping_sensitivity.py",
                       "exp02_noise_count_stress.py",
                       "exp04_k_sensitivity.py"):
            print("running {} ...".format(script))
            completed = subprocess.run(
                [sys.executable, str(PACKAGE_ROOT / "experiments" / script)],
                cwd=str(PACKAGE_ROOT),
            )
            if completed.returncode != 0:
                print("  failed with exit code {}".format(completed.returncode))
                return 1

    missing = [
        path for path in (
            RESULTS / "exp01_mapping_sensitivity" / "mapping_sensitivity.csv",
            RESULTS / "exp02_noise_count_stress" / "noise_count_stress_summary.csv",
            RESULTS / "exp04_k_sensitivity" / "k_sensitivity_single_run.csv",
        ) if not path.exists()
    ]
    if missing:
        print("missing result files; run with --run or run the experiments first:")
        for path in missing:
            print("  {}".format(path))
        return 1

    checker = Checker()
    check_exp01(checker)
    check_exp02(checker)
    check_exp04(checker)

    print("exp01 + exp02 + exp04: {} of {} values match, {} mismatches".format(
        checker.total - checker.failed, checker.total, checker.failed))
    return 0 if checker.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
