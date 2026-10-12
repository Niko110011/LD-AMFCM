"""
Output handling: result directories, CSV tables, figures and the environment
report.

Every experiment writes into ``results/<experiment>/`` and ``figures/<experiment>/``
and drops an ``environment.json`` next to its tables, so that the software and
hardware versions travel with the numbers.
"""

import json
import platform
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PACKAGE_ROOT / "results"
FIGURES_DIR = PACKAGE_ROOT / "figures"


def ensure_dir(path):
    """Create ``path`` (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_dir(experiment):
    return ensure_dir(RESULTS_DIR / experiment)


def figure_dir(experiment):
    return ensure_dir(FIGURES_DIR / experiment)


def mean_std_str(mean, std, digits=4):
    """Format a mean and a standard deviation as ``mean +- std``."""
    return "{:.{d}f} +/- {:.{d}f}".format(mean, std, d=digits)


def save_table(df, experiment, filename, index=False):
    """Write a DataFrame to ``results/<experiment>/<filename>``."""
    path = result_dir(experiment) / filename
    df.to_csv(path, index=index)
    print("  saved: {}".format(path))
    return path


def save_figure(fig, experiment, filename, dpi=300):
    """Write a matplotlib figure to ``figures/<experiment>/<filename>``."""
    path = figure_dir(experiment) / filename
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    print("  saved: {}".format(path))
    return path


def configure_matplotlib():
    """Use a non-interactive backend so that the scripts run headless."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# ----------------------------------------------------------------------
# Environment report
# ----------------------------------------------------------------------

def environment_report():
    """Collect the software and hardware versions that the numbers depend on."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.replace("\n", " "),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": None,
        "packages": {},
    }

    try:
        import multiprocessing
        report["cpu_count"] = multiprocessing.cpu_count()
    except Exception:
        pass

    for name in ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "seaborn"):
        try:
            module = __import__(name)
            report["packages"][name] = getattr(module, "__version__", "unknown")
        except ImportError:
            report["packages"][name] = "not installed"

    return report


def write_environment_report(experiment, extra=None):
    """Write ``environment.json`` for an experiment and print a short summary."""
    report = environment_report()
    if extra:
        report["experiment"] = extra

    path = result_dir(experiment) / "environment.json"
    with open(str(path), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    packages = ", ".join(
        "{}={}".format(k, v) for k, v in sorted(report["packages"].items())
    )
    print("  environment: Python {} | {} | cpus={}".format(
        report["python_version"], packages, report["cpu_count"]
    ))
    print("  saved: {}".format(path))
    return path


def describe_seed_protocol(seed_spec, local_data_rng=False):
    """Human-readable summary of a seed protocol, for the run log."""
    return (
        "protocol={protocol} | data_seed={data} | init_seed={init} | "
        "runs={n} | data_rng={rng}".format(
            protocol=("regenerated-input" if seed_spec.regenerate_data
                      else "frozen-input"),
            data=seed_spec.data_seed,
            init=seed_spec.init_seed,
            n=seed_spec.n_runs,
            rng=("per-benchmark RandomState" if local_data_rng else "shared stream"),
        )
    )
