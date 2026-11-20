"""
agents/communication.py — Discrete communication channel for MARL (C4, C5, C6).

Terminology / shapes (N = num_agents, K = num_messages, B = batch size):
  • logits   : (B, K)   — raw scores from the message head
  • message  : (B, K)   — one-hot float (straight-through in training, argmax in eval)
  • msg_bank : (B, N, K) — all agents' messages for one time step
  • received : (B, N-1, K) — what agent i sees (all messages except its own)

Key design decisions
--------------------
1.  MessageHead.forward() uses Gumbel-Softmax (hard=True, ST estimator) during
    training so the sender's logits receive gradients from the receiver's TD loss.
2.  MessageEmbedding uses nn.Linear (not nn.Embedding) over the continuous one-hot
    vectors so the gradient path sender-logits → receiver-Q is unbroken.
3.  Random-message control (C6) samples discrete uniform one-hots with no
    parameters and no gradient flow.
4.  Hand-designed baseline encodes coarse quadrant + nearest-target direction as a
    deterministic symbol in {0, …, K-1}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Message Head  (learns to generate discrete symbols)
# ---------------------------------------------------------------------------

class MessageHead(nn.Module):
    """
    Maps an agent's hidden/feature state → discrete one-hot message of size K.

    Training  : straight-through Gumbel-Softmax with temperature τ.
    Evaluation: deterministic argmax → one-hot (no randomness).

    Args:
        input_dim   (int)   : dimensionality of the agent's state vector
        num_messages (int)  : vocabulary size K  (default 4; supports 2, 4, 8, …)
        tau         (float) : initial Gumbel-Softmax temperature (default 1.0)
    """

    def __init__(
        self,
        input_dim: int,
        num_messages: int = 4,
        tau: float = 1.0,
    ):
        super(MessageHead, self).__init__()
        self.num_messages = num_messages
        self.tau = tau
        self.message_head = nn.Linear(input_dim, num_messages)

    def forward(self, state: torch.Tensor):
        """
        Args:
            state : FloatTensor (B, input_dim)

        Returns:
            message : FloatTensor (B, K) — one-hot (straight-through or argmax)
            logits  : FloatTensor (B, K) — raw pre-softmax scores (for entropy / MI)
        """
        logits = self.message_head(state)

        if self.training:
            # Straight-through Gumbel-Softmax — discrete in forward, smooth in backward
            message = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
        else:
            # Deterministic evaluation: argmax → one-hot
            idx = logits.argmax(dim=-1)                         # (B,)
            message = F.one_hot(idx, num_classes=self.num_messages).float()

        return message, logits

    def anneal_tau(self, factor: float = 0.995, min_tau: float = 0.1):
        """Multiplicative temperature annealing (call once per episode or update)."""
        self.tau = max(self.tau * factor, min_tau)


# Backward-compat alias for the original stub name
class CommunicationNetwork(MessageHead):
    """
    Backward-compatible alias for MessageHead.

    Original stub signature:
        CommunicationNetwork(input_dim, hidden_dim=128, num_messages=4)

    The ``hidden_dim`` argument is accepted but ignored — the linear head maps
    directly from ``input_dim`` to ``num_messages`` (matching the original stub).
    The forward() signature is also preserved:
        forward(encoded_observation, training=True) → (message, logits)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,   # accepted for compat; not used
        num_messages: int = 4,
        tau: float = 1.0,
    ):
        super(CommunicationNetwork, self).__init__(
            input_dim=input_dim,
            num_messages=num_messages,
            tau=tau,
        )

    def forward(self, encoded_observation: torch.Tensor, training: bool = True):
        """
        Compat shim: switches train/eval mode on the fly if the caller passes
        ``training`` explicitly (old API).  New callers should rely on
        module.train() / module.eval() instead.
        """
        was_training = self.training
        if training:
            self.train()
        else:
            self.eval()
        out = super().forward(encoded_observation)
        # Restore original mode
        if was_training:
            self.train()
        else:
            self.eval()
        return out


# ---------------------------------------------------------------------------
# 2.  Message Embedding  (receiver side — gradient-transparent)
# ---------------------------------------------------------------------------

class MessageEmbedding(nn.Module):
    """
    Embeds received one-hot messages into a dense vector.

    Uses nn.Linear (NOT nn.Embedding) so that gradients from the receiver's
    Q-network flow back through the continuous Gumbel-Softmax relaxation to the
    sender's MessageHead logits.

    Args:
        num_messages (int) : vocabulary size K
        embed_dim    (int) : output embedding dimension (default = K)
    """

    def __init__(self, num_messages: int = 4, embed_dim: int = None):
        super(MessageEmbedding, self).__init__()
        self.num_messages = num_messages
        self.embed_dim = embed_dim or num_messages
        self.linear = nn.Linear(num_messages, self.embed_dim, bias=False)

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """
        Embed a tensor of one-hot messages via a shared linear projection.

        Args:
            messages : FloatTensor (..., K)  — one-hot or ST-Gumbel vectors

        Returns:
            embedded : FloatTensor (..., embed_dim)
        """
        return self.linear(messages)

    def embed_received(self, received: torch.Tensor) -> torch.Tensor:
        """
        Embed and flatten the (N-1) received messages into a single vector
        ready for concatenation with [obs, agent_id] in DQNAgent.

        Args:
            received : FloatTensor
                (N-1, K)      — single agent, unbatched (inference)
                (B, N-1, K)   — batched replay

        Returns:
            flat : FloatTensor
                ((N-1) * embed_dim,)       when input is (N-1, K)
                (B, (N-1) * embed_dim)     when input is (B, N-1, K)

        Usage (Person 2 wiring)::

            # Build comm input for DQNAgent at each step:
            received = aggregate_messages_batched(msg_bank, receiver_id=i)  # (B, N-1, K)
            comm_flat = msg_embed.embed_received(received)                   # (B, (N-1)*embed_dim)
            net_input = torch.cat([obs_flat, agent_id, comm_flat], dim=-1)
        """
        batched = (received.dim() == 3)
        embedded = self.linear(received)          # (..., N-1, embed_dim)
        if batched:
            B = embedded.shape[0]
            return embedded.reshape(B, -1)        # (B, (N-1)*embed_dim)
        else:
            return embedded.reshape(-1)           # ((N-1)*embed_dim,)


# ---------------------------------------------------------------------------
# 3.  Broadcast / Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_messages(
    messages: torch.Tensor,
    receiver_id: int,
) -> torch.Tensor:
    """
    Extract all messages except the receiver's own message.

    Args:
        messages    : FloatTensor (N, K) — one message per agent (unbatched)
        receiver_id : int                — index of the receiving agent

    Returns:
        received : FloatTensor (N-1, K) — ordered [0, …, i-1, i+1, …, N-1]
    """
    N = messages.shape[0]
    idx = [j for j in range(N) if j != receiver_id]
    return messages[idx]                              # (N-1, K)


def aggregate_messages_batched(
    messages: torch.Tensor,
    receiver_id: int,
) -> torch.Tensor:
    """
    Batched version: extract all messages except receiver's.

    Args:
        messages    : FloatTensor (B, N, K)
        receiver_id : int

    Returns:
        received : FloatTensor (B, N-1, K)
    """
    N = messages.shape[1]
    idx = [j for j in range(N) if j != receiver_id]
    return messages[:, idx, :]                        # (B, N-1, K)


def zero_messages(
    num_agents: int,
    num_messages: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Return a zero (silent) message bank for t=0 initialisation.

    Returns:
        FloatTensor (N, num_messages) of zeros
    """
    return torch.zeros(num_agents, num_messages, device=device)


def zero_messages_batched(
    batch_size: int,
    num_agents: int,
    num_messages: int,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Batched zero message bank.

    Returns:
        FloatTensor (B, N, num_messages) of zeros
    """
    return torch.zeros(batch_size, num_agents, num_messages, device=device)


# ---------------------------------------------------------------------------
# 4.  Random-Message Control  (C6)
# ---------------------------------------------------------------------------

class RandomCommunication(nn.Module):
    """
    Produces uniform-random discrete one-hot messages with shape (B, K).
    No learnable parameters.  No gradients flow through this module.

    Args:
        num_messages (int) : vocabulary size K
    """

    def __init__(self, num_messages: int = 4):
        super(RandomCommunication, self).__init__()
        self.num_messages = num_messages

    def forward(
        self,
        state: torch.Tensor,
    ):
        """
        Args:
            state : FloatTensor (B, *)  — used only to infer batch size & device

        Returns:
            message : FloatTensor (B, K) — one-hot, uniform random, no grad
            logits  : None               — no logits for random baseline
        """
        B = state.shape[0]
        idx = torch.randint(0, self.num_messages, (B,), device=state.device)
        message = F.one_hot(idx, num_classes=self.num_messages).float()
        # Detach explicitly — no gradient should flow through C6 messages
        return message.detach(), None


# ---------------------------------------------------------------------------
# 5.  Hand-Designed (Heuristic) Baseline
# ---------------------------------------------------------------------------

class HeuristicCommunication(nn.Module):
    """
    Deterministic rule-based message encoder.

    Encodes a sender's coarse quadrant position (4 quadrants on a grid) and
    the direction to the nearest visible target into a discrete symbol in
    {0, ..., K-1} returned as a one-hot vector of size K.

    **Environment coordinate convention** (matches grid_world.py / observations.py):
      - ``agent_pos`` can be supplied as either:
        (a) Raw integer grid coordinates ``(row, col)`` in ``[0, grid_size-1]``, or
        (b) Normalised ``pbar`` values in ``[0, 1]`` as returned by
            ``get_all_observations()`` (shape ``(N, 2)``).
      - Both formats are detected automatically via ``pos.max() > 1.0``.
      - Row 0 = top of the grid; Col 0 = left of the grid.

    Encoding scheme:
      K=2 : top half (rows 0..4) → 0, bottom half (rows 5..9) → 1
      K=4 : quadrant id ∈ {0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right}
      K=8 : quadrant_id * 2  +  (1 if any target in same quadrant else 0)
      K≠2,4,8 : uniform row-quantisation into K bands

    All outputs are deterministic and gradient-free.

    Args:
        num_messages (int) : vocabulary size K  (2, 4, or 8 recommended)
        grid_size    (int) : side length of the grid (default 10)
    """

    def __init__(self, num_messages: int = 4, grid_size: int = 10):
        super(HeuristicCommunication, self).__init__()
        self.num_messages = num_messages
        self.grid_size = grid_size

    def encode(
        self,
        agent_pos: torch.Tensor,
        target_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute symbol indices from raw grid positions.

        Args:
            agent_pos  : FloatTensor (B, 2) — normalised [0,1] row/col or raw ints
            target_pos : FloatTensor (B, T, 2) or None — nearest visible targets

        Returns:
            symbols : LongTensor (B,)  in {0, …, K-1}
        """
        B = agent_pos.shape[0]
        G = self.grid_size
        K = self.num_messages

        # Normalise if positions appear to be raw grid integers
        pos = agent_pos.float()
        if pos.max() > 1.0:
            pos = pos / max(G - 1, 1)

        # Quadrant: 2×2 grid split → ids 0,1,2,3
        row_half = (pos[:, 0] >= 0.5).long()   # 0=top,  1=bottom
        col_half = (pos[:, 1] >= 0.5).long()   # 0=left, 1=right
        quadrant = row_half * 2 + col_half     # {0,1,2,3}

        if K == 2:
            symbols = row_half                  # just top/bottom
        elif K == 4:
            symbols = quadrant
        elif K == 8:
            # Combine quadrant (3 bits covers 0-7 → use quadrant*2 + target_flag)
            if target_pos is not None:
                # Simple "target visible" flag: any target in the same quadrant
                target_flag = torch.zeros(B, dtype=torch.long, device=agent_pos.device)
                for b in range(B):
                    tp = target_pos[b].float()
                    if tp.max() > 1.0:
                        tp = tp / max(G - 1, 1)
                    t_row = (tp[:, 0] >= 0.5).long()
                    t_col = (tp[:, 1] >= 0.5).long()
                    t_quad = t_row * 2 + t_col
                    if (t_quad == quadrant[b]).any():
                        target_flag[b] = 1
                symbols = (quadrant * 2 + target_flag).clamp(0, K - 1)
            else:
                symbols = (quadrant * 2).clamp(0, K - 1)
        else:
            # Fallback: uniform quantisation of row position
            symbols = (pos[:, 0] * K).long().clamp(0, K - 1)

        return symbols

    def forward(
        self,
        state: torch.Tensor,
        agent_pos: torch.Tensor = None,
        target_pos: torch.Tensor = None,
    ):
        """
        Args:
            state     : FloatTensor (B, *)  — used for device / batch size; can be
                        a GRU hidden or raw observation
            agent_pos : FloatTensor (B, 2)  — raw or normalised grid position;
                        if None, falls back to first two dims of state
            target_pos: FloatTensor (B, T, 2) or None

        Returns:
            message : FloatTensor (B, K) — deterministic one-hot, no grad
            logits  : None
        """
        if agent_pos is None:
            # Fallback: use the first two values of the state as a proxy for position
            agent_pos = state[:, :2]

        symbols = self.encode(agent_pos, target_pos)
        message = F.one_hot(symbols, num_classes=self.num_messages).float().detach()
        return message, None


# ---------------------------------------------------------------------------
# 6.  Diagnostic Utilities
# ---------------------------------------------------------------------------

def message_entropy(
    logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute batch-averaged message symbol entropy from raw logits.

    High entropy ≈ diverse messages; low entropy ≈ message collapse.

    Args:
        logits : FloatTensor (B, K) — raw (unnormalised) message logits
        eps    : small constant for numerical stability

    Returns:
        entropy : scalar FloatTensor — mean entropy over the batch (nats)
    """
    probs = F.softmax(logits, dim=-1).clamp(min=eps)
    ent = -(probs * probs.log()).sum(dim=-1)   # (B,)
    return ent.mean()


def message_entropy_from_onehots(
    messages: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate empirical symbol entropy from a collection of one-hot messages.

    Args:
        messages : FloatTensor (B, K) — one-hot vectors
        eps      : small constant for numerical stability

    Returns:
        entropy : scalar FloatTensor — empirical marginal entropy (nats)
    """
    # Empirical symbol frequency over batch
    freq = messages.mean(dim=0).clamp(min=eps)   # (K,)
    return -(freq * freq.log()).sum()
