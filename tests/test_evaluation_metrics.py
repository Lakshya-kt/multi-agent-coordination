"""
tests/test_evaluation_metrics.py — Unit tests for evaluation/metrics.py.
"""

import math
import numpy as np
import pytest

from evaluation.metrics import (
    calculate_success_rate,
    calculate_mean_completion_time,
    calculate_mean_return,
    calculate_recovery_ratio,
    calculate_interaction,
    calculate_redundant_visits,
)


def test_success_rate_basic():
    assert calculate_success_rate(successes=[True, False, True, True]) == 0.75
    assert calculate_success_rate(successes=[1, 0, 1, 0]) == 0.5
    assert calculate_success_rate(predictions=[1, 2, 3], ground_truth=[1, 2, 4]) == pytest.approx(2 / 3)


def test_success_rate_empty():
    assert calculate_success_rate([]) == 0.0
    assert calculate_success_rate(predictions=[], ground_truth=[]) == 0.0


def test_success_rate_mismatch():
    with pytest.raises(ValueError):
        calculate_success_rate(predictions=[1, 2], ground_truth=[1])


def test_mean_completion_time_basic():
    times = [10, 20, 30, 50]
    successes = [True, True, True, False]
    # Mean of 10, 20, 30 = 20.0 (excludes failed episode at 50)
    assert calculate_mean_completion_time(times, successes) == 20.0


def test_mean_completion_time_no_successes():
    times = [50, 50, 50]
    successes = [False, False, False]
    res = calculate_mean_completion_time(times, successes)
    assert math.isnan(res)


def test_mean_completion_time_empty():
    assert math.isnan(calculate_mean_completion_time([]))


def test_mean_return():
    assert calculate_mean_return([1.5, 2.5, 5.0]) == 3.0
    assert calculate_mean_return([]) == 0.0


def test_recovery_ratio_normal():
    # Full recovery: J_X = J_C1 = 0.9, J_C2 = 0.5 -> (0.9 - 0.5) / (0.9 - 0.5) = 1.0
    res = calculate_recovery_ratio(j_x=0.9, j_c2=0.5, j_c1=0.9)
    assert res["is_stable"] is True
    assert res["recovery_ratio"] == pytest.approx(1.0)
    assert res["reporting_mode"] == "ratio"

    # Half recovery: J_X = 0.7
    res2 = calculate_recovery_ratio(j_x=0.7, j_c2=0.5, j_c1=0.9)
    assert res2["recovery_ratio"] == pytest.approx(0.5)


def test_recovery_ratio_near_zero_denominator():
    # Near zero denominator: J_C1 = 0.51, J_C2 = 0.50 -> |denom| = 0.01 < 0.02
    res = calculate_recovery_ratio(j_x=0.55, j_c2=0.50, j_c1=0.51, min_denominator=0.02)
    assert res["is_stable"] is False
    assert res["reporting_mode"] == "raw_difference"
    # Returns raw diff J_X - J_C2 = 0.05
    assert res["recovery_ratio"] == pytest.approx(0.05)
    assert res["raw_difference"] == pytest.approx(0.05)


def test_interaction_categories():
    # Complementarity: (0.8 - 0.6) - (0.5 - 0.4) = 0.2 - 0.1 = +0.1 > 0.02
    comp = calculate_interaction(s_c5=0.8, s_c4=0.6, s_c3=0.5, s_c2=0.4)
    assert comp["regime"] == "complementarity"
    assert comp["delta_int"] == pytest.approx(0.1)

    # Substitution: (0.7 - 0.6) - (0.6 - 0.4) = 0.1 - 0.2 = -0.1 < -0.02
    sub = calculate_interaction(s_c5=0.7, s_c4=0.6, s_c3=0.6, s_c2=0.4)
    assert sub["regime"] == "substitution"
    assert sub["delta_int"] == pytest.approx(-0.1)

    # Additivity: (0.7 - 0.5) - (0.5 - 0.3) = 0.2 - 0.2 = 0.0
    add = calculate_interaction(s_c5=0.7, s_c4=0.5, s_c3=0.5, s_c2=0.3)
    assert add["regime"] == "additivity"
    assert abs(add["delta_int"]) < 1e-6


def test_redundant_visits():
    targets = [[2, 2], [5, 5]]
    # Step 0: agent 0 at [2, 2] (initial coverage), agent 1 at [0, 0]
    # Step 1: agent 0 stays at [2, 2] (redundant visit!), agent 1 at [5, 5] (initial coverage)
    # Step 2: agent 0 and agent 1 both at [5, 5] (both redundant!)
    trajectory = [
        [[2, 2], [0, 0]],
        [[2, 2], [5, 5]],
        [[5, 5], [5, 5]],
    ]
    res = calculate_redundant_visits(trajectory, targets)
    assert res["total_redundant_visits"] == 3
    # Total visits: step 0 has 1, step 1 has 2, step 2 has 2 -> total 5 visits
    assert res["redundant_rate"] == pytest.approx(3 / 5)
