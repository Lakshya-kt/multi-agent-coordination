"""
evaluation/evaluate.py — Comprehensive evaluation engine for C1–C6 and Heuristic Baseline.

Authoritative reference: Group_9_FAI_Project_Idea.pdf Section 3.4 & Table 2.

Features:
  1. Checkpoint loading with exact architecture reconstruction.
  2. Pure greedy action selection (epsilon=0.0).
  3. Deterministic argmax message generation at evaluation (policy_net.eval()).
  4. Precise GRU hidden-state reset and progression.
  5. Exact one-step message broadcast delay.
  6. Episode-level and step-level message/direction recording.
  7. Exporting detailed CSVs and machine-readable JSON summaries.
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from agents.dqn import DQNAgent
from environment.grid_world import GridWorld
from evaluation.baselines import GreedyHeuristicPolicy
from evaluation.metrics import (
    calculate_mean_completion_time,
    calculate_mean_return,
    calculate_success_rate,
)
from evaluation.mutual_information import (
    bootstrap_confidence_interval,
    calculate_mutual_information,
    compute_nearest_target_direction,
)


def format_obs(raw_obs: Any, num_agents: int) -> np.ndarray:
    """
    Format raw observation tuple (windows, pbar) into 2D float array (N, obs_dim).
    Matches training/train.py format convention exactly.
    """
    if isinstance(raw_obs, tuple) and len(raw_obs) == 2:
        windows, pbar = raw_obs
        w_flat = np.asarray(windows, dtype=np.float32).reshape(num_agents, -1)
        p_flat = np.asarray(pbar, dtype=np.float32).reshape(num_agents, -1)
        return np.concatenate([w_flat, p_flat], axis=-1)
    arr = np.asarray(raw_obs, dtype=np.float32)
    return arr.reshape(num_agents, -1)


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[DQNAgent, Dict[str, Any]]:
    """
    Load model weights and instantiate the exact DQNAgent matching the checkpoint config.

    Args:
        checkpoint_path: Path to saved .pt checkpoint.
        device: "cpu" or "cuda".

    Returns:
        agent: Instantiated DQNAgent set to eval() mode.
        cfg: Configuration dictionary stored in checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    dev = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=dev)
    cfg = ckpt["config"]
    obs_dim = ckpt["obs_dim"]

    agent = DQNAgent(
        input_dim=obs_dim,
        num_agents=cfg["num_agents"],
        num_actions=5,
        hidden_dim=cfg["hidden_dim"],
        lr=cfg.get("learning_rate", 5e-4),
        gamma=cfg.get("gamma", 0.99),
        batch_size=cfg.get("batch_size", 64),
        use_memory=cfg.get("use_memory", False),
        comm_mode=cfg.get("comm_mode", "none"),
        num_messages=cfg.get("num_messages", 4),
        device=dev,
    )

    agent.policy_net.load_state_dict(ckpt["policy_state_dict"])
    # CRITICAL: eval() switches MessageHead to deterministic argmax mode
    agent.policy_net.eval()
    agent.reset_hidden()
    return agent, cfg


def evaluate_agent(
    agent: DQNAgent,
    env: GridWorld,
    num_episodes: int = 200,
    seed: int = 1000,
    radius: Union[int, str] = 2,
    record_messages: bool = True,
) -> Dict[str, Any]:
    """
    Run greedy evaluation episodes on fresh randomized scenarios.

    Args:
        agent: DQNAgent in eval mode.
        env: GridWorld environment instance.
        num_episodes: Number of fresh evaluation episodes (default 200).
        seed: Base seed for reproducible evaluation episodes.
        radius: Observation radius for nearest target visibility.
        record_messages: Whether to record step-level (M, D) pairs.

    Returns:
        Comprehensive evaluation results dictionary.
    """
    episode_returns: List[float] = []
    completion_times: List[int] = []
    successes: List[bool] = []
    episode_steps: List[int] = []

    # Episode-stratified message & direction data: list of (msgs_list, dirs_list)
    episodes_md_data: List[Tuple[List[int], List[int]]] = []
    all_raw_messages: List[Dict[str, Any]] = []

    num_agents = env.num_agents

    for ep in range(num_episodes):
        ep_seed = seed + ep
        raw_obs = env.reset(seed=ep_seed)
        obs = format_obs(raw_obs, num_agents)

        agent.reset_hidden()
        prev_messages = agent.init_messages()

        ep_return = 0.0
        ep_success = False
        comp_time = env.max_steps
        step_count = 0

        ep_msgs: List[int] = []
        ep_dirs: List[int] = []

        for step in range(env.max_steps):
            step_count += 1

            # Pure greedy action selection (epsilon=0.0)
            actions, current_messages = agent.select_actions(
                observations=obs,
                epsilon=0.0,
                prev_messages=prev_messages,
                return_messages=True,
            )

            # Record message symbols M and visible target directions D
            if record_messages and current_messages is not None:
                for i in range(num_agents):
                    m_i = int(np.argmax(current_messages[i]))
                    d_i = compute_nearest_target_direction(
                        agent_pos=env.agent_position[i],
                        target_positions=env.target_position,
                        radius=radius,
                    )
                    ep_msgs.append(m_i)
                    ep_dirs.append(d_i)
                    all_raw_messages.append({
                        "episode": ep + 1,
                        "step": step_count,
                        "agent_id": i,
                        "message": m_i,
                        "direction": d_i,
                    })

            step_out = env.step(actions)
            if len(step_out) == 4:
                raw_next_obs, reward, done, info = step_out
            else:
                raw_next_obs, reward, done = step_out
                info = {}

            ep_return += float(reward)
            obs = format_obs(raw_next_obs, num_agents)
            prev_messages = current_messages

            if done:
                is_suc = bool(info.get("success", env.all_covered()))
                if is_suc:
                    ep_success = True
                    comp_time = int(info.get("t", step_count))
                break

        episode_returns.append(ep_return)
        completion_times.append(comp_time)
        successes.append(ep_success)
        episode_steps.append(step_count)
        episodes_md_data.append((ep_msgs, ep_dirs))

    # Compute high-level metrics
    s_rate = calculate_success_rate(successes=successes)
    mean_tau = calculate_mean_completion_time(completion_times, successes=successes)
    mean_ret = calculate_mean_return(episode_returns)

    # Mutual information analysis
    if record_messages and any(len(m) > 0 for m, d in episodes_md_data):
        mi_stats = bootstrap_confidence_interval(
            episodes_data=episodes_md_data,
            num_messages=agent.num_messages,
            seed=seed,
        )
    else:
        mi_stats = {
            "mi_point": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "std_err": float("nan"),
        }

    return {
        "num_episodes": num_episodes,
        "success_rate": s_rate,
        "mean_completion_time": mean_tau,
        "mean_return": mean_ret,
        "mutual_information_bits": mi_stats["mi_point"],
        "mi_ci_lower": mi_stats["ci_lower"],
        "mi_ci_upper": mi_stats["ci_upper"],
        "episode_returns": episode_returns,
        "completion_times": completion_times,
        "successes": successes,
        "episode_steps": episode_steps,
        "raw_messages": all_raw_messages,
    }


def evaluate_heuristic_baseline(
    env: GridWorld,
    num_episodes: int = 200,
    seed: int = 1000,
    radius: Union[int, str] = 2,
) -> Dict[str, Any]:
    """
    Run greedy heuristic baseline policy across fresh episodes.
    """
    policy = GreedyHeuristicPolicy(seed=seed)
    num_agents = env.num_agents

    episode_returns: List[float] = []
    completion_times: List[int] = []
    successes: List[bool] = []
    episode_steps: List[int] = []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        raw_obs = env.reset(seed=ep_seed)

        ep_return = 0.0
        ep_success = False
        comp_time = env.max_steps
        step_count = 0

        for step in range(env.max_steps):
            step_count += 1
            actions = policy.select_actions(raw_obs, radius=radius)
            raw_obs, reward, done, info = env.step(actions)
            ep_return += float(reward)

            if done:
                is_suc = bool(info.get("success", env.all_covered()))
                if is_suc:
                    ep_success = True
                    comp_time = int(info.get("t", step_count))
                break

        episode_returns.append(ep_return)
        completion_times.append(comp_time)
        successes.append(ep_success)
        episode_steps.append(step_count)

    s_rate = calculate_success_rate(successes=successes)
    mean_tau = calculate_mean_completion_time(completion_times, successes=successes)
    mean_ret = calculate_mean_return(episode_returns)

    return {
        "condition": "heuristic",
        "radius": radius,
        "num_agents": num_agents,
        "num_episodes": num_episodes,
        "success_rate": s_rate,
        "mean_completion_time": mean_tau,
        "mean_return": mean_ret,
        "mutual_information_bits": float("nan"),
        "mi_ci_lower": float("nan"),
        "mi_ci_upper": float("nan"),
        "episode_returns": episode_returns,
        "completion_times": completion_times,
        "successes": successes,
        "episode_steps": episode_steps,
    }


def save_evaluation_outputs(
    results: Dict[str, Any],
    run_id: str,
    results_dir: str = "results",
) -> Dict[str, str]:
    """
    Save evaluation artifacts:
      1. {results_dir}/{run_id}_eval_summary.json
      2. {results_dir}/{run_id}_eval_episodes.csv
      3. {results_dir}/{run_id}_eval_messages.csv (if messages recorded)
    """
    os.makedirs(results_dir, exist_ok=True)
    paths = {}

    # 1. Summary JSON
    summary_path = os.path.join(results_dir, f"{run_id}_eval_summary.json")
    summary_data = {
        "run_id": run_id,
        "condition": results.get("condition"),
        "radius": results.get("radius"),
        "num_agents": results.get("num_agents"),
        "seed": results.get("seed"),
        "num_episodes": results["num_episodes"],
        "success_rate": results["success_rate"],
        "mean_completion_time": results["mean_completion_time"],
        "mean_return": results["mean_return"],
        "mutual_information_bits": results["mutual_information_bits"],
        "mi_ci_lower": results.get("mi_ci_lower"),
        "mi_ci_upper": results.get("mi_ci_upper"),
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    paths["summary"] = summary_path

    # 2. Episode-level CSV
    ep_path = os.path.join(results_dir, f"{run_id}_eval_episodes.csv")
    with open(ep_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "success", "completion_time", "steps"])
        for ep in range(results["num_episodes"]):
            writer.writerow([
                ep + 1,
                round(results["episode_returns"][ep], 4),
                int(results["successes"][ep]),
                results["completion_times"][ep],
                results["episode_steps"][ep],
            ])
    paths["episodes"] = ep_path

    # 3. Message-level CSV (if present)
    if "raw_messages" in results and results["raw_messages"]:
        msg_path = os.path.join(results_dir, f"{run_id}_eval_messages.csv")
        with open(msg_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "step", "agent_id", "message_symbol", "target_direction"])
            for m in results["raw_messages"]:
                writer.writerow([
                    m["episode"],
                    m["step"],
                    m["agent_id"],
                    m["message"],
                    m["direction"],
                ])
        paths["messages"] = msg_path

    return paths


def evaluate_checkpoint(
    checkpoint_path: str,
    num_episodes: int = 200,
    results_dir: str = "results",
    eval_seed: int = 10000,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint, evaluate on fresh episodes, save artifacts, and return results.
    """
    agent, cfg = load_checkpoint(checkpoint_path, device=device)
    radius = cfg["observation_radius"]

    env = GridWorld(
        grid_size=cfg.get("grid_size", 10),
        num_agents=cfg.get("num_agents", 4),
        num_targets=cfg.get("num_targets", 4),
        max_steps=cfg.get("max_steps", 50),
        radius=radius,
    )

    results = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=num_episodes,
        seed=eval_seed,
        radius=radius,
        record_messages=(cfg.get("comm_mode", "none") != "none"),
    )

    results["condition"] = cfg.get("condition")
    results["radius"] = radius
    results["num_agents"] = cfg.get("num_agents")
    results["seed"] = cfg.get("seed")

    run_id = os.path.splitext(os.path.basename(checkpoint_path))[0]
    paths = save_evaluation_outputs(results, run_id=run_id, results_dir=results_dir)
    results["saved_paths"] = paths
    return results


def evaluate_conditions(
    checkpoints_dir: str = "checkpoints",
    results_dir: str = "results",
    num_episodes: int = 200,
    eval_seed: int = 10000,
) -> List[Dict[str, Any]]:
    """
    Discover all checkpoints in checkpoints_dir and run evaluation on each.
    """
    if not os.path.exists(checkpoints_dir):
        print(f"Directory '{checkpoints_dir}' does not exist.")
        return []

    ckpt_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
    if not ckpt_files:
        print(f"No .pt checkpoints found in {checkpoints_dir}.")
        return []

    all_results = []
    print(f"Discovered {len(ckpt_files)} checkpoints in {checkpoints_dir}.")
    for ckpt_name in sorted(ckpt_files):
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        print(f"Evaluating {ckpt_name} over {num_episodes} fresh episodes...")
        res = evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            num_episodes=num_episodes,
            results_dir=results_dir,
            eval_seed=eval_seed,
        )
        all_results.append(res)
        print(
            f"  -> Success: {res['success_rate']:.3f} | "
            f"Mean Time: {res['mean_completion_time']:.2f} | "
            f"Return: {res['mean_return']:.2f} | "
            f"MI: {res['mutual_information_bits']:.4f} bits"
        )

    # Save aggregated master summary
    master_summary_path = os.path.join(results_dir, "master_evaluation_summary.json")
    clean_summary = [
        {
            "condition": r.get("condition"),
            "radius": r.get("radius"),
            "num_agents": r.get("num_agents"),
            "seed": r.get("seed"),
            "num_episodes": r["num_episodes"],
            "success_rate": r["success_rate"],
            "mean_completion_time": r["mean_completion_time"],
            "mean_return": r["mean_return"],
            "mutual_information_bits": r["mutual_information_bits"],
            "mi_ci_lower": r.get("mi_ci_lower"),
            "mi_ci_upper": r.get("mi_ci_upper"),
        }
        for r in all_results
    ]
    with open(master_summary_path, "w") as f:
        json.dump(clean_summary, f, indent=2)

    print(f"Master evaluation summary saved to {master_summary_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoints or baselines.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to single checkpoint")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Directory of checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save eval outputs")
    parser.add_argument("--episodes", type=int, default=200, help="Number of evaluation episodes per seed")
    parser.add_argument("--eval_seed", type=int, default=10000, help="Base seed for fresh evaluation episodes")
    args = parser.parse_args()

    if args.checkpoint is not None:
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            results_dir=args.results_dir,
            eval_seed=args.eval_seed,
        )
    else:
        evaluate_conditions(
            checkpoints_dir=args.checkpoints_dir,
            results_dir=args.results_dir,
            num_episodes=args.episodes,
            eval_seed=args.eval_seed,
        )
