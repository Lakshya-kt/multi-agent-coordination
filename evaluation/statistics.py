"""
evaluation/statistics.py — Rigorous statistical analysis and hypothesis testing.

Authoritative reference: Group_9_FAI_Project_Idea.pdf Section 3.4 & 3.5.

Covers:
  - Non-parametric bootstrap confidence intervals (95% CIs) across seeds
  - Paired difference tests across matched seeds (C4 vs C6, C5 vs C3)
  - Memory-communication interaction statistical inference
  - Safe handling of missing seeds / failed runs
  - Summary aggregation for publication tables
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


def compute_mean_and_ci(
    values: Sequence[float],
    ci: float = 0.95,
    num_bootstraps: int = 2000,
    seed: Optional[int] = 42,
) -> Dict[str, float]:
    """
    Compute sample mean, standard error, and 95% non-parametric bootstrap CI.

    Args:
        values: Sequence of metric values (e.g. across seeds).
        ci: Confidence level (default 0.95).
        num_bootstraps: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        dict with: mean, std, se, ci_lower, ci_upper, n
    """
    clean_vals = [v for v in values if v is not None and not np.isnan(v)]
    n = len(clean_vals)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n": 0,
        }

    arr = np.asarray(clean_vals, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se_val = std_val / np.sqrt(n) if n > 1 else 0.0

    if n == 1:
        return {
            "mean": mean_val,
            "std": 0.0,
            "se": 0.0,
            "ci_lower": mean_val,
            "ci_upper": mean_val,
            "n": 1,
        }

    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(num_bootstraps):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(resample))

    boot_means = np.asarray(boot_means)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    return {
        "mean": mean_val,
        "std": std_val,
        "se": se_val,
        "ci_lower": lower,
        "ci_upper": upper,
        "n": n,
    }


def compute_paired_comparison(
    condition_a_by_seed: Dict[int, float],
    condition_b_by_seed: Dict[int, float],
    ci: float = 0.95,
    num_bootstraps: int = 2000,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Perform seed-matched paired difference analysis between condition A and B.

    Matches seeds common to both dictionaries. Handles missing seeds safely.

    Args:
        condition_a_by_seed: dict mapping seed -> metric value (e.g. C4)
        condition_b_by_seed: dict mapping seed -> metric value (e.g. C6)

    Returns:
        dict with:
          - "matched_seeds": list of matched seeds
          - "mean_difference": mean(A - B)
          - "ci_lower": 95% bootstrap lower bound
          - "ci_upper": 95% bootstrap upper bound
          - "p_value": bootstrap p-value for H0: difference <= 0
          - "is_significant": bool (CI does not contain 0)
    """
    common_seeds = sorted(set(condition_a_by_seed.keys()) & set(condition_b_by_seed.keys()))
    if not common_seeds:
        return {
            "matched_seeds": [],
            "mean_difference": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "is_significant": False,
        }

    vals_a = [condition_a_by_seed[s] for s in common_seeds]
    vals_b = [condition_b_by_seed[s] for s in common_seeds]
    diffs = np.asarray(vals_a, dtype=np.float64) - np.asarray(vals_b, dtype=np.float64)

    mean_diff = float(np.mean(diffs))
    n = len(diffs)

    if n == 1:
        return {
            "matched_seeds": common_seeds,
            "mean_difference": mean_diff,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "p_value": 1.0,
            "is_significant": False,
        }

    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(num_bootstraps):
        resample = rng.choice(diffs, size=n, replace=True)
        boot_diffs.append(np.mean(resample))

    boot_diffs = np.asarray(boot_diffs)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_diffs, alpha * 100))
    upper = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))

    # Significance: CI strictly excludes 0
    is_significant = (lower > 0.0) or (upper < 0.0)

    # Empirical two-sided p-value
    if mean_diff >= 0:
        p_val = 2.0 * float(np.mean(boot_diffs <= 0.0))
    else:
        p_val = 2.0 * float(np.mean(boot_diffs >= 0.0))
    p_val = float(np.clip(p_val, 0.0, 1.0))

    return {
        "matched_seeds": common_seeds,
        "mean_difference": mean_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p_val,
        "is_significant": is_significant,
    }


def aggregate_experiment_results(
    eval_records: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Aggregate evaluation records across conditions, radii, and seeds.

    Args:
        eval_records: List of evaluation record dicts.

    Returns:
        pd.DataFrame containing summary statistics (mean +/- 95% CI) per (condition, radius).
    """
    if not eval_records:
        return pd.DataFrame()

    df = pd.DataFrame(eval_records)
    group_cols = [c for c in ["condition", "radius", "num_agents"] if c in df.columns]

    rows = []
    for group_key, grp in df.groupby(group_cols):
        cond, rad, *rest = group_key if isinstance(group_key, tuple) else (group_key, None)
        n_agents = rest[0] if rest else 4

        row = {
            "condition": cond,
            "radius": rad,
            "num_agents": n_agents,
            "num_seeds": len(grp),
        }

        # Compute stats for numeric metrics
        for metric in ["success_rate", "mean_completion_time", "mean_return", "mutual_information_bits"]:
            if metric in grp.columns:
                stats = compute_mean_and_ci(grp[metric].dropna().tolist())
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_ci_lower"] = stats["ci_lower"]
                row[f"{metric}_ci_upper"] = stats["ci_upper"]
                row[f"{metric}_std"] = stats["std"]

        rows.append(row)

    return pd.DataFrame(rows)
