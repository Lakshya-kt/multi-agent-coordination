"""
tests/test_baselines.py — Unit tests for evaluation/baselines.py (Greedy Heuristic Baseline).
"""

import numpy as np
import pytest

from environment.grid_world import GridWorld
from evaluation.baselines import GreedyHeuristicPolicy


def test_baseline_stays_when_on_target():
    policy = GreedyHeuristicPolicy()
    # 5x5 window (r=2)
    window = np.zeros((3, 5, 5), dtype=np.float32)
    # Center is at (2, 2)
    window[0, 2, 2] = 1.0  # target at center
    act = policy.select_agent_action(window, radius=2)
    assert act == 4  # Stay


def test_baseline_moves_toward_visible_unassigned_target():
    policy = GreedyHeuristicPolicy()
    window = np.zeros((3, 5, 5), dtype=np.float32)
    # Target at (0, 2) -> strictly North of center (2, 2)
    window[0, 0, 2] = 1.0
    act = policy.select_agent_action(window, radius=2)
    assert act == 0  # Up


def test_baseline_avoids_target_occupied_by_other_agent():
    policy = GreedyHeuristicPolicy()
    window = np.zeros((3, 5, 5), dtype=np.float32)
    # Target 1 at (0, 2): occupied by another agent!
    window[0, 0, 2] = 1.0
    window[1, 0, 2] = 1.0

    # Target 2 at (2, 4): unoccupied!
    window[0, 2, 4] = 1.0

    act = policy.select_agent_action(window, radius=2)
    # Should choose to move towards unoccupied Target 2 (Right = 3) instead of Target 1 (Up = 0)
    assert act == 3  # Right


def test_baseline_avoids_stepping_into_walls():
    policy = GreedyHeuristicPolicy(seed=42)
    window = np.zeros((3, 5, 5), dtype=np.float32)
    # Target is North at (0, 2), but wall blocks step North at (1, 2)
    window[0, 0, 2] = 1.0
    window[2, 1, 2] = 1.0  # Wall

    act = policy.select_agent_action(window, radius=2)
    # Cannot move Up (0) because it's a wall. Must choose another open valid move.
    assert act != 0
    assert act in [1, 2, 3, 4]


def test_baseline_full_rollout_in_gridworld():
    env = GridWorld(grid_size=10, num_agents=4, num_targets=4, max_steps=50, radius=2)
    raw_obs = env.reset(seed=123)
    policy = GreedyHeuristicPolicy(seed=123)

    total_reward = 0.0
    for _ in range(50):
        actions = policy.select_actions(raw_obs, radius=2)
        assert actions.shape == (4,)
        assert ((actions >= 0) & (actions < 5)).all()
        raw_obs, reward, done, info = env.step(actions)
        total_reward += reward
        if done:
            break

    assert info["t"] <= 50
    assert total_reward > -50.0  # Finished valid trajectory
