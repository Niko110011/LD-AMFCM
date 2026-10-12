"""
Run the experiment suite.

Each experiment runs in its own process.  That isolation is deliberate: the
experiments that reproduce the original notebook's global-generator behaviour
would otherwise contaminate each other's random state.

    python run_all.py                       # everything, full grids
    python run_all.py --quick               # reduced grids, a few minutes
    python run_all.py --experiments exp06   # a subset
    python run_all.py --list                # show the experiment order

Results land in ``results/<experiment>/`` and ``figures/<experiment>/``.
A per-experiment ``environment.json`` records the software and hardware
versions used for that run.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"

EXPERIMENTS = [
    ("exp01", "exp01_mapping_sensitivity.py",
     "Sensitivity to alpha, beta and gamma"),
    ("exp02", "exp02_noise_count_stress.py",
     "Noise-count stress test"),
    ("exp03", "exp03_scalability.py",
     "Scalability in sample size and dimensionality"),
    ("exp04", "exp04_k_sensitivity.py",
     "Sensitivity to the neighbourhood size k"),
    ("exp05", "exp05_fuzzifier_sensitivity.py",
     "Sensitivity to m_min and m_max (solver variant)"),
    ("exp06", "exp06_baseline_comparison.py",
     "Baseline comparison, main table"),
    ("exp07", "exp07_density_reliability.py",
     "Density-reliability stress test"),
    ("exp08", "exp08_ablation.py",
     "Ablation of fuzzifier and weights"),
    ("exp09", "exp09_noise_ratio.py",
     "Contamination-ratio sensitivity"),
    ("exp10", "exp10_runtime.py",
     "Preprocessing and iteration cost"),
    ("exp11", "exp11_paired_statistics.py",
     "Repeated runs and significance testing"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="reduced grids and repetitions, for a fast check")
    parser.add_argument("--experiments", default=None,
                        help="comma-separated subset, e.g. exp01,exp06")
    parser.add_argument("--list", action="store_true",
                        help="list the experiments and exit")
    args = parser.parse_args()

    if args.list:
        for key, script, description in EXPERIMENTS:
            print("{:<7s} {:<40s} {}".format(key, script, description))
        return 0

    selected = None
    if args.experiments:
        selected = {item.strip() for item in args.experiments.split(",")}
        unknown = selected - {key for key, _, _ in EXPERIMENTS}
        if unknown:
            parser.error("unknown experiment(s): {}".format(
                ", ".join(sorted(unknown))))

    print("=" * 78)
    print("LD-AMFCM reproducibility package")
    print("=" * 78)
    print("mode      : {}".format("quick" if args.quick else "full"))
    print("results   : {}".format(PACKAGE_ROOT / "results"))
    print("figures   : {}".format(PACKAGE_ROOT / "figures"))
    print("")

    failures = []
    started = time.perf_counter()

    for key, script, description in EXPERIMENTS:
        if selected is not None and key not in selected:
            continue

        command = [sys.executable, str(EXPERIMENTS_DIR / script)]
        if args.quick:
            command.append("--quick")

        print("-" * 78)
        print("{}  {}".format(key, description))
        print("-" * 78)

        experiment_started = time.perf_counter()
        completed = subprocess.run(command, cwd=str(PACKAGE_ROOT))
        elapsed = time.perf_counter() - experiment_started

        if completed.returncode != 0:
            failures.append(key)
            print("  FAILED (exit code {}) after {:.1f} s".format(
                completed.returncode, elapsed))
        else:
            print("  ok, {:.1f} s".format(elapsed))
        print("")

    print("=" * 78)
    print("total wall-clock time: {:.1f} s".format(time.perf_counter() - started))
    if failures:
        print("FAILED experiments: {}".format(", ".join(failures)))
        return 1
    print("all selected experiments completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
