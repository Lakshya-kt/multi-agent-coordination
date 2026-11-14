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
    Ensures observations have shape (num_agents, obs_dim) regardless of how
    Person 1 structures GridWorld.reset() / GridWorld.step() returns.
    """
    arr = np.asarray(raw_obs, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == num_agents:
        return arr
    return arr.reshape(num_agents, -1)


def train(cfg=None):
    if cfg is None:
        cfg = get_config()

    set_seed(cfg["seed"])

    # Pass observation_radius if Person 1's GridWorld accepts it; fallback cleanly if not
    try:
        env = GridWorld(
            grid_size=cfg["grid_size"],
            num_agents=cfg["num_agents"],
            num_targets=cfg["num_targets"],
            max_steps=cfg["max_steps"],
            observation_radius=cfg["observation_radius"],
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

            ep_return = 0.0
            ep_loss = []
            steps = 0
            success = 0

            for _ in range(cfg["max_steps"]):
                steps += 1
                global_step += 1

                actions = agent.select_actions(obs, epsilon=epsilon)
                step_out = env.step(actions)

                # Support both 4-tuple (obs, reward, done, info) and 3-tuple (obs, reward, done)
                if len(step_out) == 4:
                    raw_next_obs, reward, done, info = step_out
                else:
                    raw_next_obs, reward, done = step_out
                    info = {}

                next_obs = _format_obs(raw_next_obs, env, cfg["num_agents"])
                replay_buffer.add((obs, eye_agents, actions, float(reward), next_obs, bool(done)))

                obs = next_obs
                ep_return += float(reward)

                if (
                    replay_buffer.len() >= cfg["min_replay_size"]
                    and global_step % cfg["train_frequency"] == 0
                ):
                    loss = agent.train_step(replay_buffer)
                    if loss is not None:
                        ep_loss.append(loss)

                if global_step % cfg["target_update_frequency"] == 0:
                    agent.update_target_network()

                if done:
                    if isinstance(info, dict) and "success" in info:
                        success = int(info["success"])
                    else:
                        try:
                            success = int(env.get_covered_targets() == cfg["num_targets"])
                        except Exception:
                            success = 0
                    break

            epsilon = max(cfg["epsilon_min"], epsilon * cfg["epsilon_decay"])
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
                    f"Success: {avg_suc:.2f} | Return: {avg_ret:.2f} | Eps: {epsilon:.3f}"
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