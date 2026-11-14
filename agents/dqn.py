import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_actions=5):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, observations):
        return self.network(observations)


class DQNAgent:
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
    ):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Total input concatenates local/global observation + one-hot agent ID
        total_input_dim = input_dim + num_agents

        self.policy_net = DQN(
            total_input_dim, hidden_dim=hidden_dim, num_actions=num_actions
        ).to(self.device)
        self.target_net = DQN(
            total_input_dim, hidden_dim=hidden_dim, num_actions=num_actions
        ).to(self.device)

        self.update_target_network()
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.agent_ids_eye = torch.eye(num_agents, device=self.device)

    def _concat_agent_ids(self, obs_tensor, agent_ids_tensor=None):
        if agent_ids_tensor is None:
            agent_ids_tensor = self.agent_ids_eye
        return torch.cat([obs_tensor, agent_ids_tensor], dim=-1)

    def select_actions(self, observations, epsilon=0.1):
        """
        Epsilon-greedy action selection for all N agents simultaneously.
        Args:
            observations: np.ndarray of shape (num_agents, obs_dim)
            epsilon: exploration probability
        Returns:
            actions: np.ndarray of shape (num_agents,) with ints in [0, num_actions-1]
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions, size=self.num_agents)

        with torch.no_grad():
            obs_t = torch.as_tensor(
                observations, dtype=torch.float32, device=self.device
            )
            net_in = self._concat_agent_ids(obs_t)
            q_values = self.policy_net(net_in)
            return q_values.argmax(dim=-1).cpu().numpy()

    def train_step(self, replay_buffer):
        """
        Single parameter-shared DQN update step from ReplayBuffer.
        Returns scalar loss float (or None if buffer has fewer samples than batch_size).
        """
        if replay_buffer.len() < self.batch_size:
            return None

        obs_b, ids_b, act_b, rew_b, next_obs_b, done_b = replay_buffer.sample(
            self.batch_size
        )

        B, N, obs_dim = obs_b.shape

        obs_t = torch.as_tensor(obs_b, device=self.device).reshape(B * N, obs_dim)
        ids_t = torch.as_tensor(ids_b, device=self.device).reshape(B * N, N)
        act_t = torch.as_tensor(act_b, device=self.device).reshape(B * N)
        next_obs_t = torch.as_tensor(next_obs_b, device=self.device).reshape(
            B * N, obs_dim
        )

        # Broadcast shared team reward and done flag across all N agents
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

        # Current Q(s, a)
        q_all = self.policy_net(self._concat_agent_ids(obs_t, ids_t))
        q_taken = q_all.gather(1, act_t.unsqueeze(1)).squeeze(1)

        # Bellman target using target network
        with torch.no_grad():
            next_q_all = self.target_net(self._concat_agent_ids(next_obs_t, ids_t))
            next_q_max = next_q_all.max(dim=1)[0]
            td_target = rew_t + self.gamma * (1.0 - done_t) * next_q_max

        loss = F.mse_loss(q_taken, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        """Copy the weights from the policy network to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())