import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Multi-agent replay buffer supporting both standard 1-step transition replay
    (for C1, C2, C4, C6) and sequence/trajectory replay (for C3, C5 with GRU).
    """

    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.episodes = deque(maxlen=max(100, capacity // 50))

    def add(self, experience):
        """
        Store one experience tuple in the buffer:
        (obs, agent_ids, actions, reward, next_obs, done)
        or
        (obs, agent_ids, actions, reward, next_obs, done, prev_messages, next_messages)
        where:
          obs / next_obs: (num_agents, obs_dim)
          agent_ids: (num_agents, num_agents)
          actions: (num_agents,)
          reward: float
          done: bool
          prev_messages / next_messages: (num_agents, num_messages) or None
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of multi-agent transitions."""
        batch = random.sample(self.buffer, batch_size)
        first = batch[0]
        if len(first) >= 8:
            obs, agent_ids, actions, rewards, next_obs, dones, prev_msgs, next_msgs = zip(*batch)
            prev_arr = None if prev_msgs[0] is None else np.asarray(prev_msgs, dtype=np.float32)
            next_arr = None if next_msgs[0] is None else np.asarray(next_msgs, dtype=np.float32)
            return (
                np.asarray(obs, dtype=np.float32),        # (B, N, obs_dim)
                np.asarray(agent_ids, dtype=np.float32),  # (B, N, N)
                np.asarray(actions, dtype=np.int64),      # (B, N)
                np.asarray(rewards, dtype=np.float32),    # (B,)
                np.asarray(next_obs, dtype=np.float32),   # (B, N, obs_dim)
                np.asarray(dones, dtype=np.float32),      # (B,)
                prev_arr,                                 # (B, N, K) or None
                next_arr,                                 # (B, N, K) or None
            )
        else:
            obs, agent_ids, actions, rewards, next_obs, dones = zip(*batch)
            return (
                np.asarray(obs, dtype=np.float32),        # (B, N, obs_dim)
                np.asarray(agent_ids, dtype=np.float32),  # (B, N, N)
                np.asarray(actions, dtype=np.int64),      # (B, N)
                np.asarray(rewards, dtype=np.float32),    # (B,)
                np.asarray(next_obs, dtype=np.float32),   # (B, N, obs_dim)
                np.asarray(dones, dtype=np.float32),      # (B,)
            )

    def add_episode(self, episode):
        """Store a full episode (list of experience tuples)."""
        if len(episode) > 0:
            self.episodes.append(episode)

    def sample_sequences(self, batch_size):
        """
        Sample a batch of episode trajectories padded to the max length in the batch.
        Returns:
            obs: (B, L, N, obs_dim)
            agent_ids: (B, L, N, N)
            actions: (B, L, N)
            rewards: (B, L)
            next_obs: (B, L, N, obs_dim)
            dones: (B, L)
            mask: (B, L) with 1.0 for valid steps, 0.0 for padding
            prev_messages: (B, L, N, K) or None
            next_messages: (B, L, N, K) or None
        """
        batch_eps = random.sample(self.episodes, batch_size)
        max_len = max(len(ep) for ep in batch_eps)

        sample_item = batch_eps[0][0]
        has_comm = len(sample_item) >= 8 and sample_item[6] is not None

        N, obs_dim = sample_item[0].shape
        K = sample_item[6].shape[-1] if has_comm else 0

        obs_pad = np.zeros((batch_size, max_len, N, obs_dim), dtype=np.float32)
        ids_pad = np.zeros((batch_size, max_len, N, N), dtype=np.float32)
        act_pad = np.zeros((batch_size, max_len, N), dtype=np.int64)
        rew_pad = np.zeros((batch_size, max_len), dtype=np.float32)
        next_obs_pad = np.zeros((batch_size, max_len, N, obs_dim), dtype=np.float32)
        done_pad = np.zeros((batch_size, max_len), dtype=np.float32)
        mask_pad = np.zeros((batch_size, max_len), dtype=np.float32)

        prev_msgs_pad = np.zeros((batch_size, max_len, N, K), dtype=np.float32) if has_comm else None
        next_msgs_pad = np.zeros((batch_size, max_len, N, K), dtype=np.float32) if has_comm else None

        for b, ep in enumerate(batch_eps):
            ep_len = len(ep)
            for t, trans in enumerate(ep):
                obs_pad[b, t] = trans[0]
                ids_pad[b, t] = trans[1]
                act_pad[b, t] = trans[2]
                rew_pad[b, t] = trans[3]
                next_obs_pad[b, t] = trans[4]
                done_pad[b, t] = float(trans[5])
                mask_pad[b, t] = 1.0
                if has_comm:
                    prev_msgs_pad[b, t] = trans[6]
                    next_msgs_pad[b, t] = trans[7]

        return (
            obs_pad,
            ids_pad,
            act_pad,
            rew_pad,
            next_obs_pad,
            done_pad,
            mask_pad,
            prev_msgs_pad,
            next_msgs_pad,
        )

    def num_episodes(self):
        return len(self.episodes)

    def len(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)


EpisodeReplayBuffer = ReplayBuffer
