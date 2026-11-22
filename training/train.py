import argparse
import csv
import os
import random
import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer
from environment.grid_world import GridWorld
from training.config import get_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_obs(raw_obs, env, num_agents):
    """
    Flatten the (windows, pbar) tuple from GridWorld into a 2-D float array
    of shape (num_agents, obs_dim) suitable for DQNAgent.

    Handles two return formats:
      - Tuple (windows, pbar): windows (N, C, W, W) + pbar (N, 2) → concat & flatten
      - Plain ndarray of shape (N, obs_dim) or ravellable to it
    """
    if isinstance(raw_obs, tuple) and len(raw_obs) == 2:
        windows, pbar = raw_obs
        windows = np.asarray(windows, dtype=np.float32).reshape(num_agents, -1)
        pbar = np.asarray(pbar, dtype=np.float32).reshape(num_agents, -1)
        return np.concatenate([windows, pbar], axis=-1)
    arr = np.asarray(raw_obs, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == num_agents:
        return arr
    return arr.reshape(num_agents, -1)

def _per_agent_state(env):
    """Return (on_target: (N,) float, min_dist_to_uncovered: (N,) float)."""
    N = env.num_agents
    targets = env.target_position
    agents = env.agent_position

    # Check which targets are currently occupied by at least one agent
    occupied = set(tuple(p) for p in agents)
    uncovered = np.array([t for t in targets if tuple(t) not in occupied])

    on_target = np.zeros(N, dtype=np.float32)
    min_dist = np.zeros(N, dtype=np.float32)
    target_set = set(tuple(t) for t in targets)

    for i in range(N):
        pos_i = tuple(agents[i])
        if pos_i in target_set:
            on_target[i] = 1.0
            min_dist[i] = 0.0
        elif len(uncovered) > 0:
            min_dist[i] = float(np.abs(uncovered - agents[i]).sum(axis=1).min())
    return on_target, min_dist

def train(cfg=None):
    if cfg is None:
        cfg = get_config()

    set_seed(cfg["seed"])

    # GridWorld uses 'radius' kwarg; fall back gracefully if the signature changes.
    try:
        env = GridWorld(
            grid_size=cfg["grid_size"],
            num_agents=cfg["num_agents"],
            num_targets=cfg["num_targets"],
            max_steps=cfg["max_steps"],
            radius=cfg["observation_radius"],
        )
    except TypeError:
        env = GridWorld(
            grid_size=cfg["grid_size"],
            num_agents=cfg["num_agents"],
            num_targets=cfg["num_targets"],
            max_steps=cfg["max_steps"],
        )

    init_raw = env.reset(seed=cfg["seed"])
    if init_raw is None:
        raise RuntimeError(
            "GridWorld.reset() returned None. Person 1 must finish environment/grid_world.py first."
        )

    init_obs = _format_obs(init_raw, env, cfg["num_agents"])
    obs_dim = init_obs.shape[-1]

    agent = DQNAgent(
        input_dim=obs_dim,
        num_agents=cfg["num_agents"],
        num_actions=5,
        hidden_dim=cfg["hidden_dim"],
        lr=cfg["learning_rate"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        use_memory=cfg.get("use_memory", False),
        comm_mode=cfg.get("comm_mode", "none"),
        num_messages=cfg.get("num_messages", 4),
    )
    replay_buffer = ReplayBuffer(capacity=cfg["replay_capacity"])

    eye_agents = np.eye(cfg["num_agents"], dtype=np.float32)
    epsilon = cfg["epsilon"]
    global_step = 0

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    run_id = (
        f"{cfg['condition']}_r{cfg['observation_radius']}_"
        f"N{cfg['num_agents']}_seed{cfg['seed']}"
    )
    log_path = os.path.join(cfg["results_dir"], f"{run_id}_train.csv")

    recent_returns = []
    recent_success = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "success", "steps", "epsilon", "loss"])

        for ep in range(1, cfg["num_episodes"] + 1):
            raw_obs = env.reset()
            obs = _format_obs(raw_obs, env, cfg["num_agents"])
            agent.reset_hidden()
            prev_messages = agent.init_messages()
            episode_transitions = []

            ep_return = 0.0
            ep_loss = []
            steps = 0
            success = 0

            prev_on, prev_dist = _per_agent_state(env)

            for _ in range(cfg["max_steps"]):
                steps += 1
                global_step += 1

                actions = agent.select_actions(obs, epsilon=epsilon, prev_messages=prev_messages)
                current_messages = agent.current_messages
                step_out = env.step(actions)

                if len(step_out) == 4:
                    raw_next_obs, reward, done, info = step_out
                else:
                    raw_next_obs, reward, done = step_out
                    info = {}

                next_obs = _format_obs(raw_next_obs, env, cfg["num_agents"])


                # Only zero out Bellman bootstrap on true task completion, not step-50 timeouts
                terminated = bool(info.get("success", False)) if isinstance(info, dict) else False

                curr_on, curr_dist = _per_agent_state(env)
                # +1.0 for reaching target, -1.0 for leaving, +0.15 continuous holding reward, +0.15 distance progress
                per_agent_rew = (
                    1.0 * (curr_on - prev_on)
                    + 0.15 * curr_on
                    + 0.15 * (prev_dist - curr_dist)
                    - 0.02
                )
                if terminated:
                    per_agent_rew += 10.0
                prev_on, prev_dist = curr_on, curr_dist

                transition = (
                    obs,
                    eye_agents,
                    actions,
                    per_agent_rew,
                    next_obs,
                    terminated,
                    prev_messages,
                    current_messages,
                )
                replay_buffer.add(transition)
                if cfg.get("use_memory", False):
                    episode_transitions.append(transition)

                obs = next_obs
                prev_messages = current_messages
                ep_return += float(reward)

                ready_to_train = (
                    replay_buffer.num_episodes() >= max(4, min(cfg["batch_size"] // 4, 16))
                    if cfg.get("use_memory", False)
                    else replay_buffer.len() >= cfg["min_replay_size"]
                )

                if (
                    ready_to_train
                    and global_step % cfg["train_frequency"] == 0
                ):
                    loss = agent.train_step(replay_buffer)
                    if loss is not None:
                        ep_loss.append(loss)

                if global_step % cfg["target_update_frequency"] == 0:
                    agent.update_target_network()

                if done:
                    success = int(terminated)
                    break

            if cfg.get("use_memory", False):
                replay_buffer.add_episode(episode_transitions)

            agent.anneal_tau()

            # Linear epsilon decay over 70% of training episodes
            decay_steps = max(1, int(0.7 * cfg["num_episodes"]))
            frac = min(1.0, ep / decay_steps)
            epsilon = cfg["epsilon"] + frac * (cfg["epsilon_min"] - cfg["epsilon"])
            recent_returns.append(ep_return)
            recent_success.append(success)

            mean_loss = float(np.mean(ep_loss)) if ep_loss else 0.0
            writer.writerow(
                [ep, round(ep_return, 4), success, steps, round(epsilon, 4), round(mean_loss, 5)]
            )

            if ep % cfg["log_interval"] == 0:
                avg_ret = np.mean(recent_returns[-cfg["log_interval"] :])
                avg_suc = np.mean(recent_success[-cfg["log_interval"] :])
                print(
                    f"[{run_id}] Ep {ep}/{cfg['num_episodes']} | "
                    f"Success: {avg_suc:.2f} | Return: {avg_ret:.2f} | Eps: {epsilon:.3f}",
                    flush=True,
                )

    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{run_id}.pt")
    torch.save(
        {
            "policy_state_dict": agent.policy_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "config": cfg,
            "obs_dim": obs_dim,
        },
        ckpt_path,
    )
    print(f"Training complete. Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="C1")
    parser.add_argument("--radius", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(
        condition=args.condition,
        observation_radius=args.radius,
        seed=args.seed,
        yaml_path=args.config,
    )
    train(cfg)