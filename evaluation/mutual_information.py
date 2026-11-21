"""
evaluation/mutual_information.py — Mutual information analysis between message
symbols M and nearest visible target direction D.

Authoritative reference: Group_9_FAI_Project_Idea.pdf Section 3.4 & 3.5.

Key definitions:
  - M in {0, ..., K-1}: Discrete message symbol (K=4 in main experiment)
  - D in {0, 1, 2, 3, 4}: Direction of nearest visible target
      0 = Up    (dy < 0)
      1 = Down  (dy > 0)
      2 = Left  (dx < 0)
      3 = Right (dx > 0)
      4 = None  (no target visible within observation radius r)
  - Theoretical upper bound: MI(M; D) <= min{log2(K), log2(5)} = log2(4) = 2.0 bits.
  - Bootstrap resampling operates at the independent episode unit (never step-level).
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


DIRECTION_NAMES = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
    4: "None",
}


def compute_nearest_target_direction(
    agent_pos: Sequence[int],
    target_positions: Sequence[Sequence[int]],
    radius: Union[int, str] = "full",
    covered_targets: Optional[Sequence[Sequence[int]]] = None,
) -> int:
    """
    Extract the direction category D in {0, 1, 2, 3, 4} of the nearest visible target.

    Visibility rules:
      - For integer radius r: target (yt, xt) is visible to agent (ya, xa) iff
        max(|yt - ya|, |xt - xa|) <= r  (Chebyshev / L_inf distance <= r).
      - For radius == "full": all targets are visible.
      - If no target is visible: returns 4 (None).

    Tie-breaking:
      - Distance metric: Manhattan distance |yt - ya| + |xt - xa|.
      - If an agent is already standing on a target (dist = 0) and other visible
        targets exist, prioritize the closest non-zero distance target.
      - If multiple targets have identical distance: deterministic tie-break by
        lowest target coordinate tuple (yt, xt).
      - Direction mapping from displacement (dy = yt - ya, dx = xt - xa):
        * If |dy| > |dx|: 0 (Up) if dy < 0 else 1 (Down)
        * If |dx| > |dy|: 2 (Left) if dx < 0 else 3 (Right)
        * If |dy| == |dx| > 0: vertical takes precedence: 0 (Up) if dy < 0 else 1 (Down)
        * If dy == 0 and dx == 0: returns 0 if no other target, else checked above.

    Args:
        agent_pos: (ya, xa) integer coordinates of the agent.
        target_positions: (num_targets, 2) array-like of target coordinates.
        radius: observation radius in {1, 2, 3, 5, "full"}.
        covered_targets: Optional set/list of already-covered target tuples.

    Returns:
        int in {0, 1, 2, 3, 4}.
    """
    ya, xa = int(agent_pos[0]), int(agent_pos[1])
    is_full = str(radius) == "full"
    r = 999 if is_full else int(radius)

    # Filter visible targets within Chebyshev window
    visible_targets = []
    for t in target_positions:
        yt, xt = int(t[0]), int(t[1])
        if max(abs(yt - ya), abs(xt - xa)) <= r:
            visible_targets.append((yt, xt))

    if not visible_targets:
        return 4  # No visible target

    # If standing on a target but other targets are visible within radius,
    # prioritize the nearest unreached visible target
    if len(visible_targets) > 1 and (ya, xa) in visible_targets:
        unreached = [t for t in visible_targets if t != (ya, xa)]
        if unreached:
            visible_targets = unreached

    # Sort visible targets deterministically by:
    # 1. Manhattan distance
    # 2. Coordinate tuple (yt, xt) for stable tie-breaking
    visible_targets.sort(key=lambda t: (abs(t[0] - ya) + abs(t[1] - xa), t[0], t[1]))
    nearest_yt, nearest_xt = visible_targets[0]

    dy = nearest_yt - ya
    dx = nearest_xt - xa

    if dy == 0 and dx == 0:
        return 0  # Agent on target

    if abs(dy) > abs(dx):
        return 0 if dy < 0 else 1
    elif abs(dx) > abs(dy):
        return 2 if dx < 0 else 3
    else:
        # Diagonal tie |dy| == |dx| > 0 -> deterministic vertical precedence
        return 0 if dy < 0 else 1


def calculate_mutual_information(
    messages: Sequence[int],
    directions: Sequence[int],
    num_messages: int = 4,
    num_directions: int = 5,
    eps: float = 1e-12,
) -> float:
    """
    Calculate empirical mutual information I(M; D) in bits (log base 2).

    Group_9_FAI_Project_Idea.pdf Eq. (11):
        I(M; D) = sum_{m} sum_{d} p(m, d) * log2( p(m, d) / (p(m) * p(d)) )

    Upper bound:
        I(M; D) <= min{log2(num_messages), log2(num_directions)}.
        With K=4 and 5 directions, I(M; D) <= log2(4) = 2.0 bits.

    Args:
        messages: Array/list of discrete message indices M in {0, ..., num_messages-1}.
        directions: Array/list of direction indices D in {0, ..., num_directions-1}.
        num_messages: Alphabet size K (default 4).
        num_directions: Number of direction categories (default 5).
        eps: Small float to prevent log2(0).

    Returns:
        float: Empirical mutual information in bits >= 0.0.
    """
    if len(messages) == 0 or len(directions) == 0:
        return 0.0
    if len(messages) != len(directions):
        raise ValueError(f"Length mismatch: len(messages)={len(messages)} != len(directions)={len(directions)}")

    m_arr = np.asarray(messages, dtype=np.int64)
    d_arr = np.asarray(directions, dtype=np.int64)
    N = len(m_arr)

    # 2D Joint histogram
    joint_counts = np.zeros((num_messages, num_directions), dtype=np.float64)
    for m, d in zip(m_arr, d_arr):
        if 0 <= m < num_messages and 0 <= d < num_directions:
            joint_counts[m, d] += 1.0

    p_md = joint_counts / max(N, 1)
    p_m = np.sum(p_md, axis=1, keepdims=True)  # (K, 1)
    p_d = np.sum(p_md, axis=0, keepdims=True)  # (1, 5)

    # Non-zero joint probability elements
    valid = p_md > eps
    denom = p_m @ p_d  # outer product (K, 5)

    # I(M; D) = sum p(m,d) * log2(p(m,d) / (p(m)*p(d)))
    ratio = np.where(valid, p_md / np.maximum(denom, eps), 1.0)
    mi = np.sum(np.where(valid, p_md * np.log2(np.maximum(ratio, eps)), 0.0))

    # Bound: MI must be non-negative and bounded by min(log2(K), log2(D))
    max_bound = min(np.log2(num_messages), np.log2(num_directions))
    return float(np.clip(mi, 0.0, max_bound))


def bootstrap_confidence_interval(
    episodes_data: Sequence[Tuple[Sequence[int], Sequence[int]]],
    num_bootstraps: int = 1000,
    ci: float = 0.95,
    num_messages: int = 4,
    seed: Optional[int] = 42,
) -> Dict[str, float]:
    """
    Compute non-parametric bootstrap confidence interval for Mutual Information.

    CRITICAL METHODOLOGY REQUIREMENT:
      Resampling operates at the independent EPISODE unit, not individual time steps.
      Each item in `episodes_data` represents one episode: (episode_msgs, episode_dirs).

    Args:
        episodes_data: List of episode pairs (msgs_list, dirs_list).
        num_bootstraps: Number of bootstrap resamples (default 1000).
        ci: Confidence level (default 0.95 -> [2.5%, 97.5%]).
        num_messages: K vocabulary size.
        seed: Random seed for bootstrap reproducibility.

    Returns:
        dict with:
          - "mi_point": float (estimate on full dataset)
          - "ci_lower": float (2.5 percentile)
          - "ci_upper": float (97.5 percentile)
          - "std_err": float (standard error of bootstrap distribution)
    """
    if len(episodes_data) == 0:
        return {"mi_point": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_err": 0.0}

    # Full sample estimate
    all_msgs = []
    all_dirs = []
    for m_ep, d_ep in episodes_data:
        all_msgs.extend(m_ep)
        all_dirs.extend(d_ep)

    point_mi = calculate_mutual_information(all_msgs, all_dirs, num_messages=num_messages)

    if len(episodes_data) == 1:
        return {"mi_point": point_mi, "ci_lower": point_mi, "ci_upper": point_mi, "std_err": 0.0}

    rng = np.random.default_rng(seed)
    n_ep = len(episodes_data)
    boot_mis = []

    for _ in range(num_bootstraps):
        sample_indices = rng.integers(0, n_ep, size=n_ep)
        sample_msgs = []
        sample_dirs = []
        for idx in sample_indices:
            m_ep, d_ep = episodes_data[idx]
            sample_msgs.extend(m_ep)
            sample_dirs.extend(d_ep)

        boot_mi = calculate_mutual_information(sample_msgs, sample_dirs, num_messages=num_messages)
        boot_mis.append(boot_mi)

    boot_mis = np.asarray(boot_mis)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_mis, alpha * 100))
    upper = float(np.percentile(boot_mis, (1.0 - alpha) * 100))
    std_err = float(np.std(boot_mis))

    return {
        "mi_point": float(point_mi),
        "ci_lower": lower,
        "ci_upper": upper,
        "std_err": std_err,
    }


def paired_bootstrap_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    num_bootstraps: int = 2000,
    ci: float = 0.95,
    seed: Optional[int] = 42,
) -> Dict[str, float]:
    """
    Compute paired difference across matched seeds (e.g. C4 vs C6, or C5 vs C3).

    Proposal Section 3.5:
      "C4 vs. C6 and C5 vs. C3 are paired by seed with paired bootstrap CIs
       (few seeds, so no normality assumption)."

    Args:
        values_a: Metric values across seeds for condition A (e.g. C4).
        values_b: Metric values across matched seeds for condition B (e.g. C6).
        num_bootstraps: Number of bootstrap iterations.
        ci: Confidence level.
        seed: RNG seed.

    Returns:
        dict with:
          - "mean_difference": float (mean(A - B))
          - "ci_lower": float
          - "ci_upper": float
          - "p_value_two_sided": estimated non-parametric bootstrap p-value
    """
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)

    if len(arr_a) != len(arr_b):
        raise ValueError(f"Paired comparison requires identical lengths: {len(arr_a)} != {len(arr_b)}")
    if len(arr_a) == 0:
        return {"mean_difference": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value_two_sided": 1.0}

    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_diffs = []
    for _ in range(num_bootstraps):
        resample = rng.choice(diffs, size=n, replace=True)
        boot_diffs.append(np.mean(resample))

    boot_diffs = np.asarray(boot_diffs)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_diffs, alpha * 100))
    upper = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))

    # Two-sided empirical p-value for H0: diff == 0
    # Proportion of resamples crossing 0 in opposite direction
    if mean_diff >= 0:
        p_val = 2.0 * float(np.mean(boot_diffs <= 0.0))
    else:
        p_val = 2.0 * float(np.mean(boot_diffs >= 0.0))
    p_val = float(np.clip(p_val, 0.0, 1.0))

    return {
        "mean_difference": mean_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value_two_sided": p_val,
    }
