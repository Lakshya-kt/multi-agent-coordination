# Shared experiment configuration.

CONFIG = {
    # Environment
    "grid_size": 10,
    "num_agents": 2,
    "num_targets": 2,
    "max_steps": 50,

    # Observation
    "observation_radius": 2,

    # DQN
    "hidden_dim": 128,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,

    # Communication
    "num_messages": 4,
    "message_delay": 1,

    # Training
    "num_episodes": 1000,
    "batch_size": 64,
    "replay_capacity": 10000,
    "target_update_frequency": 100,

    # Reproducibility
    "seed": 42
}

# TODO:
# Add condition-specific configurations for C1-C6.
# Add separate configurations for observation radii.