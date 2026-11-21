"""
tests/test_mutual_information.py — Unit tests for evaluation/mutual_information.py.
"""

import math
import numpy as np
import pytest

from evaluation.mutual_information import (
    compute_nearest_target_direction,
    calculate_mutual_information,
    bootstrap_confidence_interval,
    paired_bootstrap_difference,
)


def test_direction_cardinal():
    targets = [[2, 5]]
    # Agent at (5, 5) -> target is strictly North (dy = -3, dx = 0) -> Up (0)
    assert compute_nearest_target_direction(agent_pos=[5, 5], target_positions=targets, radius=5) == 0

    # Agent at (0, 5) -> target is strictly South (dy = +2, dx = 0) -> Down (1)
    assert compute_nearest_target_direction(agent_pos=[0, 5], target_positions=targets, radius=5) == 1

    # Agent at (2, 8) -> target is strictly West (dy = 0, dx = -3) -> Left (2)
    assert compute_nearest_target_direction(agent_pos=[2, 8], target_positions=targets, radius=5) == 2

    # Agent at (2, 1) -> target is strictly East (dy = 0, dx = +4) -> Right (3)
    assert compute_nearest_target_direction(agent_pos=[2, 1], target_positions=targets, radius=5) == 3


def test_direction_visibility_threshold():
    targets = [[2, 2]]
    # Agent at (2, 5) -> dist in x is 3.
    # Radius = 2: Target outside Chebyshev window -> None (4)
    assert compute_nearest_target_direction(agent_pos=[2, 5], target_positions=targets, radius=2) == 4
    # Radius = 3: Target inside window -> Left (2)
    assert compute_nearest_target_direction(agent_pos=[2, 5], target_positions=targets, radius=3) == 2
    # Radius = full: Target visible -> Left (2)
    assert compute_nearest_target_direction(agent_pos=[2, 5], target_positions=targets, radius="full") == 2


def test_direction_diagonal_tie_breaking():
    # Target at (1, 1), Agent at (3, 3) -> dy = -2, dx = -2.
    # |dy| == |dx| > 0 -> deterministic vertical precedence: Up (0)
    assert compute_nearest_target_direction(agent_pos=[3, 3], target_positions=[[1, 1]], radius=5) == 0

    # Target at (5, 5), Agent at (3, 3) -> dy = +2, dx = +2.
    # |dy| == |dx| > 0 -> deterministic vertical precedence: Down (1)
    assert compute_nearest_target_direction(agent_pos=[3, 3], target_positions=[[5, 5]], radius=5) == 1


def test_direction_standing_on_target_with_other_targets():
    # Agent at (2, 2). Target 0 at (2, 2), Target 1 at (2, 4)
    # Agent is on target 0, so should aim towards target 1 (Right = 3)
    targets = [[2, 2], [2, 4]]
    assert compute_nearest_target_direction(agent_pos=[2, 2], target_positions=targets, radius=3) == 3


def test_mutual_information_identical():
    # When M perfectly mirrors D (with K=4, map D mod 4):
    # MI should equal empirical entropy of M
    rng = np.random.default_rng(0)
    M = rng.integers(0, 4, size=1000)
    D = M.copy()
    # H(M) for 4 equally likely symbols is log2(4) = 2.0 bits
    mi = calculate_mutual_information(M, D, num_messages=4, num_directions=5)
    assert mi == pytest.approx(2.0, abs=0.05)


def test_mutual_information_independent():
    # Completely independent variables -> MI approx 0.0
    rng = np.random.default_rng(1)
    N = 5000
    M = rng.integers(0, 4, size=N)
    D = rng.integers(0, 5, size=N)
    mi = calculate_mutual_information(M, D, num_messages=4, num_directions=5)
    # Estimator bias on 5000 points is very small (< 0.02 bits)
    assert mi < 0.03


def test_mutual_information_bound():
    # Theoretical upper bound is 2.0 bits
    rng = np.random.default_rng(2)
    M = rng.integers(0, 4, size=500)
    D = rng.integers(0, 5, size=500)
    mi = calculate_mutual_information(M, D, num_messages=4, num_directions=5)
    assert 0.0 <= mi <= 2.0


def test_bootstrap_confidence_interval():
    # Generate 50 episodes of data
    rng = np.random.default_rng(3)
    episodes_data = []
    for _ in range(50):
        m_ep = rng.integers(0, 4, size=20).tolist()
        d_ep = rng.integers(0, 5, size=20).tolist()
        episodes_data.append((m_ep, d_ep))

    res = bootstrap_confidence_interval(episodes_data, num_bootstraps=200, seed=42)
    assert "mi_point" in res
    assert "ci_lower" in res
    assert "ci_upper" in res
    assert res["ci_lower"] <= res["mi_point"] <= res["ci_upper"] or abs(res["ci_lower"] - res["mi_point"]) < 1e-4


def test_paired_bootstrap_difference():
    # Condition A beats Condition B consistently across 5 seeds
    vals_a = [0.75, 0.80, 0.78, 0.82, 0.79]
    vals_b = [0.50, 0.52, 0.49, 0.55, 0.51]

    res = paired_bootstrap_difference(vals_a, vals_b, num_bootstraps=500, seed=42)
    assert res["mean_difference"] == pytest.approx(0.274, abs=0.01)
    assert res["ci_lower"] > 0.20
    assert res["ci_upper"] > res["ci_lower"]
    assert res["p_value_two_sided"] == 0.0
