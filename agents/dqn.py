import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.memory import GRUMemory
from agents.communication import (
    MessageHead,
    MessageEmbedding,
    RandomCommunication,
    aggregate_messages,
    aggregate_messages_batched,
    zero_messages,
)


class DQN(nn.Module):
    """
    Parameter-shared multi-agent Q-network supporting:
      - C1, C2: Flat observation + Agent ID -> 3-layer MLP -> Q-values
      - C3:     Flat observation + Agent ID -> GRUMemory -> Q-values
      - C4:     [Obs, ID, emb(m_{-i})] -> 3-layer MLP -> Q-values + MessageHead
      - C5:     [Obs, ID, emb(m_{-i})] -> GRUMemory -> Q-values + MessageHead
      - C6:     [Obs, ID, emb(m_{-i})] -> 3-layer MLP -> Q-values + RandomCommunication
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_actions=5,
        num_agents=4,
        use_memory=False,
        comm_mode="none",
        num_messages=4,
    ):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.use_memory = use_memory
        self.comm_mode = comm_mode
        self.num_messages = num_messages

        # Teammate communication embedding (C4, C5, C6)
        if self.comm_mode in ("learned", "random"):
            self.comm_embed = MessageEmbedding(num_messages=num_messages, embed_dim=num_messages)
            comm_in_dim = (num_agents - 1) * num_messages
        else:
            self.comm_embed = None
            comm_in_dim = 0

        # Total input dimension concatenating observation, agent ID, and communication
        self.total_input_dim = input_dim + num_agents + comm_in_dim

        # Representation trunk: GRUMemory (C3, C5) or Feedforward MLP (C1, C2, C4, C6)
        if self.use_memory:
            self.memory = GRUMemory(input_dim=self.total_input_dim, hidden_dim=hidden_dim)
            self.feature_net = None
        else:
            self.memory = None
            self.feature_net = nn.Sequential(
                nn.Linear(self.total_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        # Action Q-value head
        self.q_head = nn.Linear(hidden_dim, num_actions)

        # Message generation head
        if self.comm_mode == "learned":
            self.msg_head = MessageHead(input_dim=hidden_dim, num_messages=num_messages)
        elif self.comm_mode == "random":
            self.msg_head = RandomCommunication(num_messages=num_messages)
        else:
            self.msg_head = None

    def forward(self, x, hidden_state=None):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, total_input_dim) or (B, L, total_input_dim).
            hidden_state: Optional initial GRU hidden state for recurrent models.
        Returns:
            If use_memory or hidden_state is passed:
                q_values, feat, next_hidden
            Else:
                q_values (or tuple if comm is active)
        """
        if self.use_memory:
            feat, next_hidden = self.memory(x, hidden_state)
        else:
            feat = self.feature_net(x)
            next_hidden = None

        q_values = self.q_head(feat)

        if self.use_memory or hidden_state is not None:
            return q_values, feat, next_hidden
        return q_values


class DQNAgent:
    """
    Parameter-shared multi-agent DQN controller for conditions C1 through C6.
    """

    def __init__(
        self,
        input_dim,
        num_agents=4,
        num_actions=5,
        hidden_dim=128,
        lr=5e-4,
        gamma=0.99,
        batch_size=64,
        grad_clip=10.0,
        device=None,
        use_memory=False,
        comm_mode="none",
        num_messages=4,
    ):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.use_memory = use_memory
        self.comm_mode = comm_mode
        self.num_messages = num_messages
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = DQN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_agents=num_agents,
            use_memory=use_memory,
            comm_mode=comm_mode,
            num_messages=num_messages,
        ).to(self.device)

        self.target_net = DQN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_agents=num_agents,
            use_memory=use_memory,
            comm_mode=comm_mode,
            num_messages=num_messages,
        ).to(self.device)

        self.update_target_network()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.agent_ids_eye = torch.eye(num_agents, device=self.device)

        # Internal rollout states
        self.hidden_state = None
        self.current_messages = None
        self.prev_messages = None
        self.reset_hidden()

    def reset_hidden(self):
        """Reset recurrent hidden states and initialize zero messages for t=0."""
        self.hidden_state = None
        if self.comm_mode != "none":
            self.prev_messages = zero_messages(
                self.num_agents, self.num_messages, device=self.device
            )
            self.current_messages = self.prev_messages.clone()
        else:
            self.prev_messages = None
            self.current_messages = None

    def init_messages(self):
        """Return zero-initialized messages for episode start."""
        if self.comm_mode != "none":
            return np.zeros((self.num_agents, self.num_messages), dtype=np.float32)
        return None

    def _prepare_rollout_input(self, obs_tensor, prev_messages=None):
        """
        Concatenate [obs, agent_ids, comm_embedding] for all N agents at a single step.
        obs_tensor: (N, obs_dim)
        prev_messages: (N, num_messages) or None
        """
        ids_tensor = self.agent_ids_eye
        if self.comm_mode != "none":
            if prev_messages is None:
                prev_messages = self.prev_messages
            prev_t = torch.as_tensor(prev_messages, dtype=torch.float32, device=self.device)
            comm_list = [
                self.policy_net.comm_embed.embed_received(aggregate_messages(prev_t, i))
                for i in range(self.num_agents)
            ]
            comm_tensor = torch.stack(comm_list, dim=0)  # (N, comm_in_dim)
            return torch.cat([obs_tensor, ids_tensor, comm_tensor], dim=-1)
        return torch.cat([obs_tensor, ids_tensor], dim=-1)

    def select_actions(self, observations, epsilon=0.1, prev_messages=None, return_messages=False):
        """
        Epsilon-greedy action selection for all N agents.
        Args:
            observations: np.ndarray of shape (num_agents, obs_dim)
            epsilon: exploration probability
            prev_messages: np.ndarray of shape (num_agents, num_messages) or None
            return_messages: bool, whether to return (actions, current_messages)
        Returns:
            actions: np.ndarray of shape (num_agents,)
            (optional) current_messages: np.ndarray of shape (num_agents, num_messages)
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            net_in = self._prepare_rollout_input(obs_t, prev_messages)

            if self.use_memory:
                q_values, feat, self.hidden_state = self.policy_net(net_in, self.hidden_state)
            else:
                out = self.policy_net(net_in)
                if isinstance(out, tuple):
                    q_values, feat, _ = out
                else:
                    q_values = out
                    feat = self.policy_net.feature_net(net_in)

            # Generate messages if communication is active
            if self.comm_mode != "none":
                msgs, _ = self.policy_net.msg_head(feat)
                self.current_messages = msgs.cpu().numpy()
                self.prev_messages = msgs

            actions = q_values.argmax(dim=-1).cpu().numpy()

        if np.random.rand() < epsilon:
            actions = np.random.randint(0, self.num_actions, size=self.num_agents)

        if return_messages:
            return actions, self.current_messages
        return actions

    def _prepare_batch_comm(self, msgs_tensor, net, B):
        """
        Build comm embeddings for B*N agents given batch msgs tensor (B, N, K).
        Returns tensor of shape (B*N, (N-1)*K).
        """
        comm_list = []
        for i in range(self.num_agents):
            rec_i = aggregate_messages_batched(msgs_tensor, receiver_id=i)  # (B, N-1, K)
            flat_i = net.comm_embed.embed_received(rec_i)                   # (B, (N-1)*K)
            comm_list.append(flat_i)
        return torch.stack(comm_list, dim=1).reshape(B * self.num_agents, -1)

    def train_step(self, replay_buffer):
        """
        Perform one training update. Dispatches to transition training (C1, C2, C4, C6)
        or sequence training (C3, C5).
        """
        if self.use_memory:
            return self._train_step_sequence(replay_buffer)
        return self._train_step_transition(replay_buffer)

    def _train_step_transition(self, replay_buffer):
        """1-step transition update for C1, C2, C4, C6."""
        if replay_buffer.len() < self.batch_size:
            return None

        sample = replay_buffer.sample(self.batch_size)
        if len(sample) >= 8 and sample[6] is not None:
            obs_b, ids_b, act_b, rew_b, next_obs_b, done_b, prev_msg_b, next_msg_b = sample
        else:
            obs_b, ids_b, act_b, rew_b, next_obs_b, done_b = sample[:6]
            prev_msg_b, next_msg_b = None, None

        B, N, obs_dim = obs_b.shape

        obs_t = torch.as_tensor(obs_b, device=self.device).reshape(B * N, obs_dim)
        ids_t = torch.as_tensor(ids_b, device=self.device).reshape(B * N, N)
        act_t = torch.as_tensor(act_b, device=self.device).reshape(B * N)
        next_obs_t = torch.as_tensor(next_obs_b, device=self.device).reshape(B * N, obs_dim)

        rew_t = (
            torch.as_tensor(rew_b, device=self.device)
            .unsqueeze(1)
            .expand(B, N)
            .reshape(B * N)
        )
        done_t = (
            torch.as_tensor(done_b, device=self.device)
            .unsqueeze(1)
            .expand(B, N)
            .reshape(B * N)
        )

        if self.comm_mode != "none" and prev_msg_b is not None:
            prev_msgs_t = torch.as_tensor(prev_msg_b, device=self.device)
            next_msgs_t = torch.as_tensor(next_msg_b, device=self.device)
            comm_t = self._prepare_batch_comm(prev_msgs_t, self.policy_net, B)
            next_comm_t = self._prepare_batch_comm(next_msgs_t, self.target_net, B)
            net_in = torch.cat([obs_t, ids_t, comm_t], dim=-1)
            next_net_in = torch.cat([next_obs_t, ids_t, next_comm_t], dim=-1)
        else:
            net_in = torch.cat([obs_t, ids_t], dim=-1)
            next_net_in = torch.cat([next_obs_t, ids_t], dim=-1)

        # Current Q(s, a)
        out = self.policy_net(net_in)
        q_all = out[0] if isinstance(out, tuple) else out
        q_taken = q_all.gather(1, act_t.unsqueeze(1)).squeeze(1)

        # Bellman target
        with torch.no_grad():
            next_out = self.target_net(next_net_in)
            next_q_all = next_out[0] if isinstance(next_out, tuple) else next_out
            next_q_max = next_q_all.max(dim=1)[0]
            td_target = rew_t + self.gamma * (1.0 - done_t) * next_q_max

        loss = F.mse_loss(q_taken, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item())

    def _train_step_sequence(self, replay_buffer):
        """Recurrent sequence/trajectory update for C3, C5."""
        min_eps = max(4, min(self.batch_size // 4, 16))
        if replay_buffer.num_episodes() < min_eps:
            return None

        sample_b = min(self.batch_size, replay_buffer.num_episodes())
        (
            obs_b,
            ids_b,
            act_b,
            rew_b,
            next_obs_b,
            done_b,
            mask_b,
            prev_msg_b,
            next_msg_b,
        ) = replay_buffer.sample_sequences(sample_b)

        B, L, N, obs_dim = obs_b.shape

        obs_t = torch.as_tensor(obs_b, device=self.device)           # (B, L, N, obs_dim)
        ids_t = torch.as_tensor(ids_b, device=self.device)           # (B, L, N, N)
        act_t = torch.as_tensor(act_b, device=self.device)           # (B, L, N)
        rew_t = torch.as_tensor(rew_b, device=self.device)           # (B, L)
        next_obs_t = torch.as_tensor(next_obs_b, device=self.device) # (B, L, N, obs_dim)
        done_t = torch.as_tensor(done_b, device=self.device)         # (B, L)
        mask_t = torch.as_tensor(mask_b, device=self.device)         # (B, L)

        # Build sequence inputs
        if self.comm_mode != "none" and prev_msg_b is not None:
            prev_msg_t = torch.as_tensor(prev_msg_b, device=self.device) # (B, L, N, K)
            next_msg_t = torch.as_tensor(next_msg_b, device=self.device) # (B, L, N, K)

            comm_steps = []
            next_comm_steps = []
            for l in range(L):
                c_l = self._prepare_batch_comm(prev_msg_t[:, l], self.policy_net, B)
                nc_l = self._prepare_batch_comm(next_msg_t[:, l], self.target_net, B)
                comm_steps.append(c_l)
                next_comm_steps.append(nc_l)

            # (L, B*N, comm_dim) -> (B*N, L, comm_dim)
            comm_seq = torch.stack(comm_steps, dim=0).transpose(0, 1)
            next_comm_seq = torch.stack(next_comm_steps, dim=0).transpose(0, 1)

            obs_flat = obs_t.permute(0, 2, 1, 3).reshape(B * N, L, obs_dim)
            ids_flat = ids_t.permute(0, 2, 1, 3).reshape(B * N, L, N)
            next_obs_flat = next_obs_t.permute(0, 2, 1, 3).reshape(B * N, L, obs_dim)

            seq_in = torch.cat([obs_flat, ids_flat, comm_seq], dim=-1)
            next_seq_in = torch.cat([next_obs_flat, ids_flat, next_comm_seq], dim=-1)
        else:
            obs_flat = obs_t.permute(0, 2, 1, 3).reshape(B * N, L, obs_dim)
            ids_flat = ids_t.permute(0, 2, 1, 3).reshape(B * N, L, N)
            next_obs_flat = next_obs_t.permute(0, 2, 1, 3).reshape(B * N, L, obs_dim)

            seq_in = torch.cat([obs_flat, ids_flat], dim=-1)
            next_seq_in = torch.cat([next_obs_flat, ids_flat], dim=-1)

        # Unroll policy network
        h_seq, _ = self.policy_net.memory(seq_in)              # (B*N, L, hidden_dim)
        q_seq = self.policy_net.q_head(h_seq)                  # (B*N, L, num_actions)

        act_flat = act_t.permute(0, 2, 1).reshape(B * N, L)    # (B*N, L)
        q_taken = q_seq.gather(-1, act_flat.unsqueeze(-1)).squeeze(-1)  # (B*N, L)

        # Target network unroll
        with torch.no_grad():
            next_h_seq, _ = self.target_net.memory(next_seq_in)
            next_q_seq = self.target_net.q_head(next_h_seq)
            next_q_max = next_q_seq.max(dim=-1)[0]             # (B*N, L)

            rew_expanded = rew_t.unsqueeze(1).expand(B, N, L).reshape(B * N, L)
            done_expanded = done_t.unsqueeze(1).expand(B, N, L).reshape(B * N, L)
            td_target = rew_expanded + self.gamma * (1.0 - done_expanded) * next_q_max

        mask_flat = mask_t.unsqueeze(1).expand(B, N, L).reshape(B * N, L)
        td_error = (q_taken - td_target) ** 2
        loss = (td_error * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def anneal_tau(self, factor=0.995, min_tau=0.1):
        """Anneal Gumbel-Softmax temperature if learned comm is active."""
        if self.policy_net.msg_head is not None and hasattr(self.policy_net.msg_head, "anneal_tau"):
            self.policy_net.msg_head.anneal_tau(factor, min_tau)
