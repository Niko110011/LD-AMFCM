"""
Statistical protocol.

Repeated runs are compared against the FCM baseline with a paired t-test on
matched runs, complemented by Cohen's d_z for paired samples.  Multiplicity is
controlled with the Holm step-down procedure.  A Friedman test on the
dataset-level mean ARI with Kendall's W provides the omnibus test, followed by
Wilcoxon signed-rank post-hoc comparisons against the baseline.
"""

import numpy as np
from scipy import stats

SIGNIFICANCE_LEVEL = 0.05


def cohen_d_paired(x, y):
    """Cohen's d_z for paired samples."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    sd = np.std(diff, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(diff) / sd)


def effect_magnitude(d):
    """Conventional magnitude band for an absolute effect size."""
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    if d < 0.5:
        return "Small"
    if d < 0.8:
        return "Medium"
    return "Large"


def holm_correction(p_values):
    """Holm step-down adjusted p-values, order preserving."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(m)
    for i, p in enumerate(sorted_p):
        adjusted_sorted[i] = (m - i) * p

    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def paired_tests_vs_baseline(records, baseline_method, value_column="ARI",
                             method_column="Method", dataset_column="Dataset",
                             run_column="Run"):
    """Paired t-test, Cohen's d_z and Holm correction against a baseline.

    ``records`` must be a row-per-run table so that the pairing between a
    method and the baseline is explicit.
    """
    import pandas as pd

    df = pd.DataFrame.from_records(records)
    rows = []

    for dataset in df[dataset_column].unique():
        baseline = (
            df[(df[dataset_column] == dataset) &
               (df[method_column] == baseline_method)]
            .sort_values(run_column)[value_column]
            .values
        )

        for method in df[method_column].unique():
            if method == baseline_method:
                continue

            comparison = (
                df[(df[dataset_column] == dataset) &
                   (df[method_column] == method)]
                .sort_values(run_column)[value_column]
                .values
            )

            if len(comparison) != len(baseline):
                continue

            t_stat, p_value = stats.ttest_rel(comparison, baseline)
            mean_diff = float(np.mean(comparison - baseline))
            d_value = cohen_d_paired(comparison, baseline)

            rows.append({
                "Dataset": dataset,
                "Method": method,
                "Mean_Diff": mean_diff,
                "t_stat": float(t_stat),
                "p-value": float(p_value),
                "Cohen_d": d_value,
                "Magnitude": effect_magnitude(d_value),
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["Holm_p"] = holm_correction(result["p-value"].values)
        result["Sig"] = np.where(
            result["Holm_p"] < SIGNIFICANCE_LEVEL, "*", "n.s."
        )
    return result


def friedman_test(dataset_method_means, methods):
    """Friedman omnibus test on the dataset-level mean ARI, plus Kendall's W."""
    table = dataset_method_means[methods]
    statistic, p_value = stats.friedmanchisquare(
        *[table[m].values for m in methods]
    )

    n_datasets = len(table)
    n_methods = len(methods)
    kendall_w = statistic / (n_datasets * (n_methods - 1))

    return float(statistic), float(p_value), float(kendall_w)


def wilcoxon_posthoc(dataset_method_means, methods, baseline_method):
    """Wilcoxon signed-rank post-hoc tests against the baseline, Holm corrected."""
    import pandas as pd

    baseline = dataset_method_means[baseline_method].values
    rows = []

    for method in methods:
        if method == baseline_method:
            continue

        values = dataset_method_means[method].values
        w_stat, p_value = stats.wilcoxon(values, baseline)
        rows.append({
            "Method": method,
            "Mean_Diff": float(np.mean(values - baseline)),
            "Wilcoxon_stat": float(w_stat),
            "p-value": float(p_value),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["Holm_p"] = holm_correction(result["p-value"].values)
        result["Sig"] = np.where(
            result["Holm_p"] < SIGNIFICANCE_LEVEL, "*", "n.s."
        )
    return result
