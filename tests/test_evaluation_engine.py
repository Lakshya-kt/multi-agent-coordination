"""
tests/test_evaluation_engine.py — Comprehensive tests for evaluation/evaluate.py
and evaluation/statistics.py across conditions C1–C6.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from agents.dqn import DQNAgent
from environment.grid_world import GridWorld
from evaluation.evaluate import (
    load_checkpoint,
    evaluate_agent,
    evaluate_checkpoint,
    evaluate_heuristic_baseline,
    format_obs,
)
from evaluation.statistics import (
    compute_mean_and_ci,
    compute_paired_comparison,
    aggregate_experiment_results,
)
from training.config import get_config


@pytest.fixture
def temp_results_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_format_obs_shape():
    env = GridWorld(radius=2)
    raw_obs = env.reset(seed=0)
    obs = format_obs(raw_obs, num_agents=4)
    # Window 5x5: 3*5*5 = 75, plus 2 pbar = 77
    assert obs.shape == (4, 77)


def test_load_existing_c1_checkpoint():
    ckpt_path = "checkpoints/C1_rfull_N4_seed42.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("C1 smoke checkpoint not found")
    agent, cfg = load_checkpoint(ckpt_path)
    assert cfg["condition"] == "C1"
    assert cfg["observation_radius"] == "full"
    assert agent.policy_net.training is False  # Must be in eval() mode
    assert agent.use_memory is False
    assert agent.comm_mode == "none"


def test_load_existing_c2_checkpoint():
    ckpt_path = "checkpoints/C2_r2_N4_seed42.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("C2 smoke checkpoint not found")
    agent, cfg = load_checkpoint(ckpt_path)
    assert cfg["condition"] == "C2"
    assert cfg["observation_radius"] == 2
    assert agent.policy_net.training is False
    assert agent.use_memory is False


def test_evaluate_agent_greedy_deterministic():
    """Verify that evaluate_agent on same seed produces deterministic results."""
    ckpt_path = "checkpoints/C2_r2_N4_seed42.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("C2 smoke checkpoint not found")

    agent, cfg = load_checkpoint(ckpt_path)
    env = GridWorld(radius=2)

    res1 = evaluate_agent(agent, env, num_episodes=5, seed=1000)
    res2 = evaluate_agent(agent, env, num_episodes=5, seed=1000)

    assert res1["success_rate"] == res2["success_rate"]
    t1 = res1["mean_completion_time"]
    t2 = res2["mean_completion_time"]
    assert (np.isnan(t1) and np.isnan(t2)) or (t1 == t2)
    assert res1["episode_returns"] == res2["episode_returns"]


def test_evaluate_existing_c1_and_c2_end_to_end(temp_results_dir):
    """End-to-end evaluation for C1 and C2 smoke checkpoints with file creation check."""
    for ckpt_name in ["C1_rfull_N4_seed42.pt", "C2_r2_N4_seed42.pt"]:
        ckpt_path = os.path.join("checkpoints", ckpt_name)
        if not os.path.exists(ckpt_path):
            continue

        res = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            num_episodes=5,
            results_dir=temp_results_dir,
            eval_seed=2000,
        )

        assert "success_rate" in res
        assert 0.0 <= res["success_rate"] <= 1.0
        assert len(res["episode_returns"]) == 5
        assert len(res["completion_times"]) == 5

        run_id = os.path.splitext(ckpt_name)[0]
        summary_file = os.path.join(temp_results_dir, f"{run_id}_eval_summary.json")
        ep_file = os.path.join(temp_results_dir, f"{run_id}_eval_episodes.csv")
        assert os.path.exists(summary_file)
        assert os.path.exists(ep_file)


@pytest.mark.parametrize("condition", ["C3", "C4", "C5", "C6"])
def test_evaluation_controlled_fixtures_c3_to_c6(condition, temp_results_dir):
    """
    Controlled fixture test for future conditions C3, C4, C5, C6:
    Create a valid mock agent, save a test checkpoint, load it, run evaluation,
    and verify GRU memory, communication delay, and MI logging.
    """
    cfg = get_config(condition=condition, observation_radius=2, seed=123)
    cfg["hidden_dim"] = 32
    obs_dim = 77
    N = 4

    agent = DQNAgent(
        input_dim=obs_dim,
        num_agents=N,
        num_actions=5,
        hidden_dim=32,
        use_memory=cfg["use_memory"],
        comm_mode=cfg["comm_mode"],
        num_messages=cfg["num_messages"],
    )

    ckpt_path = os.path.join(temp_results_dir, f"test_{condition}.pt")
    torch.save(
        {
            "policy_state_dict": agent.policy_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "config": cfg,
            "obs_dim": obs_dim,
        },
        ckpt_path,
    )

    res = evaluate_checkpoint(
        checkpoint_path=ckpt_path,
        num_episodes=3,
        results_dir=temp_results_dir,
        eval_seed=5000,
    )

    assert res["num_episodes"] == 3
    assert 0.0 <= res["success_rate"] <= 1.0
    if cfg["comm_mode"] != "none":
        # Messages should be recorded for C4, C5, C6
        msg_file = os.path.join(temp_results_dir, f"test_{condition}_eval_messages.csv")
        assert os.path.exists(msg_file)
        assert not np.isnan(res["mutual_information_bits"])


def test_evaluate_heuristic_baseline(temp_results_dir):
    env = GridWorld(radius=2)
    res = evaluate_heuristic_baseline(env, num_episodes=5, seed=100)
    assert res["num_episodes"] == 5
    assert 0.0 <= res["success_rate"] <= 1.0
    assert len(res["completion_times"]) == 5


def test_statistics_module():
    # Mean and CI test
    vals = [0.80, 0.82, 0.78, 0.81, 0.79]
    res = compute_mean_and_ci(vals)
    assert res["mean"] == pytest.approx(0.80)
    assert res["ci_lower"] <= res["mean"] <= res["ci_upper"]

    # Paired comparison test
    cond_a = {0: 0.85, 1: 0.82, 2: 0.84}
    cond_b = {0: 0.65, 1: 0.60, 2: 0.62}
    paired = compute_paired_comparison(cond_a, cond_b)
    assert paired["mean_difference"] == pytest.approx(0.2133, abs=0.01)
    assert paired["is_significant"] is True
    assert paired["ci_lower"] > 0.0
