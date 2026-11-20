"""
agents/memory.py — GRU-based recurrent memory for DRQN-style agents (C3, C5).

Design contract:
  • Single-step inference : x  shape (B, input_dim)   → out (B, hidden_dim), h (1, B, hidden_dim)
  • Sequence training     : x  shape (B, L, input_dim) → out (B, L, hidden_dim), h (1, B, hidden_dim)
  • init_hidden(B, device) → zero tensor (1, B, hidden_dim)
  • apply_done_mask(h, done) resets rows where done==True at episode boundaries
"""

import torch
import torch.nn as nn


class GRUMemory(nn.Module):
    """
    Thin wrapper around nn.GRU that handles both single-step and sequence-batch
    inputs transparently.

    Args:
        input_dim  (int): Dimensionality of the input feature vector x_i^t.
        hidden_dim (int): Dimensionality of the GRU hidden state h_i^t.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(GRUMemory, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # batch_first=True → input/output tensors are (B, L, dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_hidden(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Return a zero-initialised hidden state.

        Returns:
            h0: FloatTensor of shape (1, batch_size, hidden_dim)
                — matches nn.GRU's (num_layers, B, H) convention.
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor = None,
    ):
        """
        Run one GRU step (or a full sequence).

        Args:
            x      : FloatTensor, either
                       (B, input_dim)      – single step (squeezed internally)
                     or
                       (B, L, input_dim)   – sequence of L steps
            hidden : FloatTensor (1, B, hidden_dim) or None.
                     If None, GRU defaults to zero hidden state.

        Returns:
            output : FloatTensor
                       (B, hidden_dim)      when x was (B, input_dim)
                       (B, L, hidden_dim)   when x was (B, L, input_dim)
            hidden : FloatTensor (1, B, hidden_dim)  — updated hidden state
        """
        single_step = (x.dim() == 2)
        if single_step:
            # Add sequence-length dimension: (B, input_dim) → (B, 1, input_dim)
            x = x.unsqueeze(1)

        output, hidden = self.gru(x, hidden)

        if single_step:
            # Remove sequence dimension: (B, 1, hidden_dim) → (B, hidden_dim)
            output = output.squeeze(1)

        return output, hidden

    # ------------------------------------------------------------------
    # Episode-boundary utilities
    # ------------------------------------------------------------------

    @staticmethod
    def apply_done_mask(
        hidden: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero out hidden states for finished episodes so the next episode
        starts with a clean slate.

        Args:
            hidden : FloatTensor (1, B, hidden_dim)
            done   : BoolTensor  (B,) — True where the episode just ended

        Returns:
            hidden : FloatTensor (1, B, hidden_dim) with reset rows
        """
        # done: (B,) → (1, B, 1)  for broadcasting
        mask = (~done).float().unsqueeze(0).unsqueeze(-1)  # (1, B, 1)
        return hidden * mask

    @staticmethod
    def reset_hidden(
        hidden: torch.Tensor,
        agent_indices=None,
    ) -> torch.Tensor:
        """
        Explicitly zero hidden state for a list of agent/batch indices.

        Args:
            hidden        : FloatTensor (1, B, hidden_dim)
            agent_indices : list[int] or None — if None, reset all

        Returns:
            hidden with selected indices zeroed out
        """
        if agent_indices is None:
            return torch.zeros_like(hidden)
        hidden = hidden.clone()
        hidden[:, agent_indices, :] = 0.0
        return hidden


# ---------------------------------------------------------------------------
# Backward-compat alias kept so any existing import of ``AgentMemory`` works.
# ---------------------------------------------------------------------------

class AgentMemory(GRUMemory):
    """
    Backward-compatible alias for GRUMemory.

    Original stub signature: AgentMemory(input_dim, hidden_dim=128)
    This class preserves that exact interface while adding the full
    GRUMemory API.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(AgentMemory, self).__init__(input_dim=input_dim, hidden_dim=hidden_dim)
