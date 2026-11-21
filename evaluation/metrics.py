"""
evaluation/metrics.py — Core evaluation metrics for multi-agent coordination.

Authoritative reference: Group_9_FAI_Project_Idea.pdf Section 3.4 & 3.5.
Calculates:
  - Success rate: S_hat_X(r)
  - Mean completion time (successful episodes only): tau_hat_X(r)
  - Mean episode return: G_hat_X(r)
  - Recovery ratio: rho_X(r) with stability thresholding
  - Memory-communication interaction: Delta_int(r)
  - Redundant target visits
"""

import math
from typing import Dict, List, Optional, Sequence, Union
import numpy as np


def calculate_success_rate(
    predictions: Optional[Sequence[Union[bool, int, float]]] = None,
    ground_truth: Optional[Sequence[Union[bool, int, float]]] = None,
    successes: Optional[Sequence[Union[bool, int, float]]] = None,
) -> float:
    """
    Calculate the success rate S_hat_X(r) over evaluation episodes.

    Args:
        predictions: Optional binary indicators or predicted labels.
        ground_truth: Optional ground truth labels (if comparing predictions vs gt).
        successes: Direct sequence of boolean/int episode success indicators (sigma_e).

    Returns:
        float: Success rate in [0.0, 1.0]. Returns 0.0 if input is empty.
    """
    data = successes if successes is not None else predictions
    if data is None or len(data) == 0:
        return 0.0

    arr = np.asarray(data)
    if ground_truth is not None:
        gt = np.asarray(ground_truth)
        if len(arr) != len(gt):
            raise ValueError(f"Length mismatch: len(predictions)={len(arr)} != len(ground_truth)={len(gt)}")
        return float(np.mean(arr == gt))

    # Binary success indicators
    return float(np.mean(arr.astype(bool)))


def calculate_mean_completion_time(
    completion_times: Sequence[Union[int, float]],
    successes: Optional[Sequence[Union[bool, int, float]]] = None,
) -> float:
    """
    Calculate mean completion time tau_hat_X(r) over successful episodes only.

    Group_9_FAI_Project_Idea.pdf Eq. (8):
        tau_hat_X(r) = (1 / |{e : sigma_e = 1}|) * sum_{e : sigma_e = 1} t_e^*

    Args:
        completion_times: Sequence of completion step counts t_e for each episode.
        successes: Optional sequence of success indicators sigma_e.
                   If None, completion_times are assumed to already be filtered.

    Returns:
        float: Mean completion time of successful episodes.
               Returns float('nan') if no successful episodes occurred or input is empty.
    """
    if len(completion_times) == 0:
        return float("nan")

    times_arr = np.asarray(completion_times, dtype=np.float64)

    if successes is not None:
        suc_arr = np.asarray(successes, dtype=bool)
        if len(times_arr) != len(suc_arr):
            raise ValueError(f"Length mismatch: {len(times_arr)} != {len(suc_arr)}")
        valid_times = times_arr[suc_arr]
    else:
        valid_times = times_arr

    if len(valid_times) == 0:
        return float("nan")

    return float(np.mean(valid_times))


def calculate_mean_return(episode_returns: Sequence[Union[int, float]]) -> float:
    """
    Calculate mean episode return G_hat_X(r).

    Group_9_FAI_Project_Idea.pdf Eq. (9):
        G_hat_X(r) = (1 / E) * sum_{e=1}^E sum_t R_e^t

    Args:
        episode_returns: List/array of undiscounted cumulative episode returns.

    Returns:
        float: Mean return. Returns 0.0 if input is empty.
    """
    if len(episode_returns) == 0:
        return 0.0
    return float(np.mean(np.asarray(episode_returns, dtype=np.float64)))


def calculate_recovery_ratio(
    j_x: float,
    j_c2: float,
    j_c1: float,
    min_denominator: float = 0.02,
) -> Dict[str, Union[float, bool, str]]:
    """
    Calculate the recovery ratio rho_X(r).

    Group_9_FAI_Project_Idea.pdf Eq. (10):
        rho_X(r) = (J_X(r) - J_C2(r)) / (J_C1 - J_C2(r))

    Where J_X is performance (success rate by default).
    rho_X = 1.0 indicates full recovery, 0.0 indicates no benefit, <0 indicates harm.

    Safety:
        When denominator magnitude |J_C1 - J_C2(r)| < min_denominator, reporting
        a ratio is numerically unstable. In that case, report the raw performance
        difference (J_X - J_C2) instead.

    Returns:
        dict with:
          - "recovery_ratio": float (rho_X if stable, else raw difference)
          - "raw_difference": float (J_X - J_C2)
          - "denominator": float (J_C1 - J_C2)
          - "is_stable": bool (True if |denominator| >= min_denominator)
          - "reporting_mode": str ("ratio" or "raw_difference")
    """
    denominator = j_c1 - j_c2
    raw_diff = j_x - j_c2

    if abs(denominator) < min_denominator:
        return {
            "recovery_ratio": float(raw_diff),
            "raw_difference": float(raw_diff),
            "denominator": float(denominator),
            "is_stable": False,
            "reporting_mode": "raw_difference",
        }

    ratio = raw_diff / denominator
    return {
        "recovery_ratio": float(ratio),
        "raw_difference": float(raw_diff),
        "denominator": float(denominator),
        "is_stable": True,
        "reporting_mode": "ratio",
    }


def calculate_interaction(
    s_c5: float,
    s_c4: float,
    s_c3: float,
    s_c2: float,
) -> Dict[str, Union[float, str]]:
    """
    Calculate memory-communication interaction Delta_int(r).

    Group_9_FAI_Project_Idea.pdf Eq. (12):
        Delta_int(r) = [S_C5(r) - S_C4(r)] - [S_C3(r) - S_C2(r)]

    Interpretation:
        Delta_int > 0: Complementarity (joint benefit > sum of individual benefits)
        Delta_int ~= 0: Additivity
        Delta_int < 0: Substitution (memory and communication substitute for each other)

    Returns:
        dict with:
          - "delta_int": float
          - "regime": str ("complementarity", "additivity", or "substitution")
    """
    delta_int = float((s_c5 - s_c4) - (s_c3 - s_c2))
    eps = 0.02
    if delta_int > eps:
        regime = "complementarity"
    elif delta_int < -eps:
        regime = "substitution"
    else:
        regime = "additivity"

    return {
        "delta_int": delta_int,
        "regime": regime,
    }


def calculate_redundant_visits(
    agent_trajectories: Sequence[Sequence[Sequence[int]]],
    target_positions: Sequence[Sequence[int]],
) -> Dict[str, Union[int, float]]:
    """
    Measure redundant target visits over one or more episodes.

    A redundant visit occurs when an agent visits a target cell that is already
    covered (either by another agent at the same step, or previously covered
    and currently occupied).

    Args:
        agent_trajectories: List of step-wise agent positions.
                            Shape: (num_steps, num_agents, 2)
        target_positions: Fixed target positions (num_targets, 2) for the episode.

    Returns:
        dict with:
          - "total_redundant_visits": int
          - "redundant_rate": float (redundant visits / total agent-target overlaps)
    """
    target_set = {tuple(t) for t in target_positions}
    total_redundant = 0
    total_target_visits = 0

    covered_set = set()
    for step_positions in agent_trajectories:
        step_covered_now = set()
        for pos in step_positions:
            pos_tup = tuple(pos)
            if pos_tup in target_set:
                total_target_visits += 1
                if pos_tup in covered_set or pos_tup in step_covered_now:
                    total_redundant += 1
                else:
                    step_covered_now.add(pos_tup)
        covered_set.update(step_covered_now)

    redundant_rate = (
        float(total_redundant / total_target_visits)
        if total_target_visits > 0
        else 0.0
    )
    return {
        "total_redundant_visits": total_redundant,
        "redundant_rate": redundant_rate,
    }
