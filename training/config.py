import copy
import yaml

# Shared base experiment configuration (matches report defaults: N=4, 10x10 grid, T=50)
CONFIG = {
    # Experiment tag
    "condition": "C1",

    # Environment
    "grid_size": 10,
    "num_agents": 4,
    "num_targets": 4,
    "max_steps": 50,

    # Observation ('full' or integer radius in {1, 2, 3, 5})
    "observation_radius": "full",

    # Architecture flags (used by Person 3 for C3-C6)
    "use_memory": False,
    "comm_mode": "none",  # "none", "learned", "random", "heuristic"

    # DQN
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9992,

    # Communication
    "num_messages": 4,
    "message_delay": 1,

    # Training
    "num_episodes": 6000,
    "batch_size": 64,
    "replay_capacity": 50000,
    "min_replay_size": 1000,
    "train_frequency": 4,
    "target_update_frequency": 250,

    # Logging & Checkpoints
    "log_interval": 200,
    "checkpoint_dir": "checkpoints",
    "results_dir": "results",

    # Reproducibility
    "seed": 0,
}

CONDITION_CONFIGS = {
    "C1": {"observation_radius": "full", "use_memory": False, "comm_mode": "none"},
    "C2": {"observation_radius": 2, "use_memory": False, "comm_mode": "none"},
    "C3": {
        "observation_radius": 2,
        "use_memory": True,
        "comm_mode": "none",
        "train_frequency": 16,
    },
    "C4": {"observation_radius": 2, "use_memory": False, "comm_mode": "learned"},
    "C5": {
        "observation_radius": 2,
        "use_memory": True,
        "comm_mode": "learned",
        "train_frequency": 16,
    },
    "C6": {"observation_radius": 2, "use_memory": False, "comm_mode": "random"},
}

OBSERVATION_RADII = [1, 2, 3, 5, "full"]


def get_config(condition="C1", observation_radius=None, seed=None, yaml_path=None):
    cfg = copy.deepcopy(CONFIG)
    if condition in CONDITION_CONFIGS:
        cfg.update(CONDITION_CONFIGS[condition])
        cfg["condition"] = condition
    if yaml_path is not None:
        with open(yaml_path, "r") as f:
            cfg.update(yaml.safe_load(f) or {})
    if observation_radius is not None:
        cfg["observation_radius"] = (
            "full" if str(observation_radius) == "full" else int(observation_radius)
        )
    if seed is not None:
        cfg["seed"] = int(seed)
    return cfg