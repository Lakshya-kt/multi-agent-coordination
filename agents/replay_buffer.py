import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        """
        Store one experience tuple in the buffer:
        (obs, agent_ids, actions, reward, next_obs, done)
        where obs/next_obs have shape (num_agents, obs_dim),
        agent_ids has shape (num_agents, num_agents),
        actions has shape (num_agents,), reward is float, done is bool.
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of multi-agent transitions and stack into numpy arrays."""
        batch = random.sample(self.buffer, batch_size)
        obs, agent_ids, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),        # (B, N, obs_dim)
            np.asarray(agent_ids, dtype=np.float32),  # (B, N, N)
            np.asarray(actions, dtype=np.int64),      # (B, N)
            np.asarray(rewards, dtype=np.float32),    # (B,)
            np.asarray(next_obs, dtype=np.float32),   # (B, N, obs_dim)
            np.asarray(dones, dtype=np.float32),      # (B,)
        )

    def len(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)