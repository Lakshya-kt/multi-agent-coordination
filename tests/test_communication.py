"""
tests/test_communication.py — Comprehensive tests for Task 3: GRU Memory & Communication.

Coverage:
  1.  GRU output shapes for N ∈ {3, 4, 6} and K ∈ {2, 4, 8}
  2.  GRU single-step vs. sequence-step equivalence
  3.  GRU hidden-state reset (done mask & explicit reset)
  4.  End-to-end gradient flow: sender_logits.grad != 0 through ST-Gumbel + Linear embed
  5.  Evaluation mode: deterministic argmax one-hot
  6.  C6 random messages: valid uniform one-hot, no grad
  7.  Heuristic baseline: deterministic valid one-hot symbols
  8.  Aggregate helpers: batched & unbatched
  9.  Diagnostic entropy helper
 10.  CommunicationNetwork backward-compat alias
 11.  AgentMemory backward-compat alias
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- imports under test ---
from agents.memory import GRUMemory, AgentMemory
from agents.communication import (
    MessageHead,
    CommunicationNetwork,
    MessageEmbedding,
    RandomCommunication,
    HeuristicCommunication,
    aggregate_messages,
    aggregate_messages_batched,
    zero_messages,
    zero_messages_batched,
    message_entropy,
    message_entropy_from_onehots,
)


# ===========================================================================
# Fixtures
# ===========================================================================

INPUT_DIM = 32
HIDDEN_DIM = 64
BATCH_SIZE = 8


@pytest.fixture(params=[3, 4, 6])
def num_agents(request):
    return request.param


@pytest.fixture(params=[2, 4, 8])
def num_messages(request):
    return request.param


@pytest.fixture
def gru():
    return GRUMemory(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)


@pytest.fixture
def device():
    return torch.device("cpu")


# ===========================================================================
# 1. GRU Output Shapes
# ===========================================================================

class TestGRUShapes:

    def test_single_step_output_shape(self, gru, device):
        """(B, input_dim) → out (B, hidden_dim), h (1, B, hidden_dim)."""
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        h0 = gru.init_hidden(BATCH_SIZE, device)
        out, h = gru(x, h0)
        assert out.shape == (BATCH_SIZE, HIDDEN_DIM), f"Expected ({BATCH_SIZE}, {HIDDEN_DIM}), got {out.shape}"
        assert h.shape == (1, BATCH_SIZE, HIDDEN_DIM), f"Expected (1, {BATCH_SIZE}, {HIDDEN_DIM}), got {h.shape}"

    def test_sequence_output_shape(self, gru, device):
        """(B, L, input_dim) → out (B, L, hidden_dim), h (1, B, hidden_dim)."""
        L = 10
        x = torch.randn(BATCH_SIZE, L, INPUT_DIM)
        h0 = gru.init_hidden(BATCH_SIZE, device)
        out, h = gru(x, h0)
        assert out.shape == (BATCH_SIZE, L, HIDDEN_DIM), f"Sequence out shape wrong: {out.shape}"
        assert h.shape == (1, BATCH_SIZE, HIDDEN_DIM)

    def test_init_hidden_shape(self, gru, device):
        h0 = gru.init_hidden(BATCH_SIZE, device)
        assert h0.shape == (1, BATCH_SIZE, HIDDEN_DIM)
        assert h0.sum().item() == 0.0, "init_hidden should return all zeros"

    def test_different_batch_sizes(self):
        gru = GRUMemory(input_dim=16, hidden_dim=32)
        for B in [1, 3, 4, 6, 16]:
            x = torch.randn(B, 16)
            out, h = gru(x)
            assert out.shape == (B, 32)
            assert h.shape == (1, B, 32)


# ===========================================================================
# 2. GRU Single-Step vs. Sequence Equivalence
# ===========================================================================

class TestGRUEquivalence:

    def test_single_vs_sequence_equivalence(self):
        """
        Running a GRU step-by-step must produce the same final hidden state
        as running the whole sequence at once.
        """
        torch.manual_seed(42)
        gru = GRUMemory(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        gru.eval()
        L = 5
        seq = torch.randn(1, L, INPUT_DIM)       # single batch item

        # --- sequence mode ---
        h0 = gru.init_hidden(1)
        seq_out, h_seq = gru(seq, h0)            # (1, L, hidden_dim)

        # --- step-by-step mode ---
        h_step = gru.init_hidden(1)
        for t in range(L):
            step_out, h_step = gru(seq[:, t, :], h_step)   # (1, input_dim)

        # Final hidden states must match
        assert torch.allclose(h_seq, h_step, atol=1e-5), (
            f"Sequence vs step-by-step hidden states differ: "
            f"max diff = {(h_seq - h_step).abs().max()}"
        )

        # Last sequence output must match last single-step output
        assert torch.allclose(seq_out[:, -1, :], step_out, atol=1e-5), (
            "Last sequence output vs last single-step output differ."
        )


# ===========================================================================
# 3. Hidden State Reset
# ===========================================================================

class TestGRUHiddenReset:

    def test_apply_done_mask_resets_rows(self):
        """Rows with done=True should be zeroed; done=False rows should be unchanged."""
        gru = GRUMemory(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        B = 4
        h = torch.ones(1, B, HIDDEN_DIM)
        done = torch.tensor([True, False, True, False])
        h_new = GRUMemory.apply_done_mask(h, done)

        # Done rows → zero
        assert h_new[0, 0].sum().item() == 0.0, "Done row 0 should be zeroed"
        assert h_new[0, 2].sum().item() == 0.0, "Done row 2 should be zeroed"
        # Not-done rows → unchanged
        assert torch.allclose(h_new[0, 1], h[0, 1]), "Non-done row 1 should be unchanged"
        assert torch.allclose(h_new[0, 3], h[0, 3]), "Non-done row 3 should be unchanged"

    def test_apply_done_mask_no_done(self):
        B = 4
        h = torch.randn(1, B, HIDDEN_DIM)
        done = torch.zeros(B, dtype=torch.bool)
        h_new = GRUMemory.apply_done_mask(h, done)
        assert torch.allclose(h_new, h), "No-done mask should leave hidden unchanged"

    def test_reset_hidden_explicit_indices(self):
        gru = GRUMemory(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        B = 6
        h = torch.ones(1, B, HIDDEN_DIM)
        h_new = GRUMemory.reset_hidden(h, agent_indices=[1, 3, 5])
        for i in range(B):
            if i in [1, 3, 5]:
                assert h_new[0, i].sum().item() == 0.0, f"Agent {i} should be reset"
            else:
                assert torch.allclose(h_new[0, i], h[0, i]), f"Agent {i} should be intact"

    def test_reset_hidden_all(self):
        gru = GRUMemory(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        h = torch.ones(1, BATCH_SIZE, HIDDEN_DIM)
        h_new = GRUMemory.reset_hidden(h, agent_indices=None)
        assert h_new.sum().item() == 0.0, "Full reset should give all zeros"


# ===========================================================================
# 4. End-to-End Gradient Flow
# ===========================================================================

class TestGradientFlow:
    """
    Critical test: verify that sender_logits.grad is non-zero when a loss
    computed from the receiver's Q-values is backpropagated through the
    Straight-Through Gumbel-Softmax message and the nn.Linear embedding.

    Chain:  sender_logits → ST-Gumbel → one-hot message
                                           ↓  (nn.Linear)
                                       embedded message
                                           ↓
                               [concat with receiver feature]
                                           ↓
                                     Q-network → loss → .backward()
    """

    def test_sender_logits_grad_through_gumbel_and_linear_embed(self):
        torch.manual_seed(0)
        K = 4
        feat_dim = 16
        embed_dim = 8
        q_hidden = 32

        # Sender produces one-hot messages
        sender_head = MessageHead(input_dim=feat_dim, num_messages=K, tau=1.0)
        sender_head.train()

        # Receiver embedding (nn.Linear — gradient-transparent)
        receiver_embed = MessageEmbedding(num_messages=K, embed_dim=embed_dim)

        # Tiny Q-network (receiver side)
        q_net = nn.Linear(feat_dim + embed_dim, 5)

        # Combine all params for a single optimizer
        all_params = (
            list(sender_head.parameters())
            + list(receiver_embed.parameters())
            + list(q_net.parameters())
        )
        optimizer = torch.optim.Adam(all_params, lr=1e-3)

        B = 4
        sender_state = torch.randn(B, feat_dim, requires_grad=True)
        receiver_state = torch.randn(B, feat_dim)

        # Forward pass
        message, logits = sender_head(sender_state)
        embedded = receiver_embed(message)                      # (B, embed_dim)
        receiver_input = torch.cat([receiver_state, embedded], dim=-1)
        q_values = q_net(receiver_input)                        # (B, 5)
        loss = q_values.mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Sender logits must have received gradients
        sender_logits_grad = sender_head.message_head.weight.grad
        assert sender_logits_grad is not None, (
            "sender_head.message_head.weight.grad is None — gradient did not flow back!"
        )
        assert sender_logits_grad.abs().max().item() > 0, (
            "sender_head.message_head.weight.grad is all zeros — gradient flow broken!"
        )

    def test_no_grad_through_random_comm(self):
        """C6: random messages must NOT carry gradients."""
        random_comm = RandomCommunication(num_messages=4)
        state = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
        message, _ = random_comm(state)
        assert not message.requires_grad, "Random messages must be detached (no grad)."

    def test_no_grad_through_heuristic(self):
        """Heuristic baseline must not carry gradients."""
        heuristic = HeuristicCommunication(num_messages=4, grid_size=10)
        state = torch.randn(BATCH_SIZE, 16)
        agent_pos = torch.randint(0, 10, (BATCH_SIZE, 2)).float()
        message, _ = heuristic(state, agent_pos=agent_pos)
        assert not message.requires_grad, "Heuristic messages must be detached."


# ===========================================================================
# 5. Evaluation Mode — Deterministic Argmax
# ===========================================================================

class TestEvalMode:

    def test_eval_mode_is_deterministic(self, num_messages):
        """In eval mode, the same input must always produce the same message."""
        head = MessageHead(input_dim=INPUT_DIM, num_messages=num_messages)
        head.eval()
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        m1, _ = head(x)
        m2, _ = head(x)
        assert torch.allclose(m1, m2), "Eval mode must be deterministic."

    def test_eval_mode_is_onehot(self, num_messages):
        """Eval mode messages must be valid one-hots (each row sums to 1, all 0/1)."""
        head = MessageHead(input_dim=INPUT_DIM, num_messages=num_messages)
        head.eval()
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        m, _ = head(x)
        assert m.shape == (BATCH_SIZE, num_messages)
        assert torch.allclose(m.sum(dim=-1), torch.ones(BATCH_SIZE)), \
            "Each eval message must sum to 1 (one-hot)."
        assert ((m == 0) | (m == 1)).all(), "Eval messages must contain only 0 or 1."

    def test_train_mode_uses_gumbel(self, num_messages):
        """In training mode, messages should still be one-hot (hard=True in ST-Gumbel)."""
        head = MessageHead(input_dim=INPUT_DIM, num_messages=num_messages)
        head.train()
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        m, _ = head(x)
        assert m.shape == (BATCH_SIZE, num_messages)
        assert torch.allclose(m.sum(dim=-1), torch.ones(BATCH_SIZE)), \
            "Train-mode messages (ST-Gumbel hard=True) must sum to 1."

    def test_train_message_has_grad(self, num_messages):
        """ST-Gumbel messages must carry a grad_fn so gradients can flow."""
        head = MessageHead(input_dim=INPUT_DIM, num_messages=num_messages)
        head.train()
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        m, logits = head(x)
        assert m.grad_fn is not None, "Training mode message must have a grad_fn."


# ===========================================================================
# 6. C6 Random Messages
# ===========================================================================

class TestRandomCommunication:

    def test_shape(self, num_agents, num_messages):
        random_comm = RandomCommunication(num_messages=num_messages)
        state = torch.randn(num_agents, INPUT_DIM)
        m, logits = random_comm(state)
        assert m.shape == (num_agents, num_messages), \
            f"Expected ({num_agents}, {num_messages}), got {m.shape}"
        assert logits is None, "Random comm should return None logits."

    def test_valid_one_hot(self, num_messages):
        random_comm = RandomCommunication(num_messages=num_messages)
        state = torch.randn(BATCH_SIZE, INPUT_DIM)
        m, _ = random_comm(state)
        assert torch.allclose(m.sum(dim=-1), torch.ones(BATCH_SIZE)), \
            "Random messages must be one-hots (sum=1)."
        assert ((m == 0) | (m == 1)).all(), "Random messages must contain only 0 or 1."

    def test_uniform_distribution(self, num_messages):
        """Over many samples, each symbol should appear roughly 1/K of the time."""
        random_comm = RandomCommunication(num_messages=num_messages)
        torch.manual_seed(123)
        B_large = 10000
        state = torch.randn(B_large, INPUT_DIM)
        m, _ = random_comm(state)
        freq = m.mean(dim=0)                       # expected ≈ 1/K per symbol
        expected = 1.0 / num_messages
        assert ((freq - expected).abs() < 0.05).all(), \
            f"Random comm not approximately uniform: freq={freq}, expected≈{expected:.3f}"

    def test_no_grad(self):
        random_comm = RandomCommunication(num_messages=4)
        state = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
        m, _ = random_comm(state)
        assert not m.requires_grad, "Random messages must be detached."


# ===========================================================================
# 7. Heuristic Baseline
# ===========================================================================

class TestHeuristicCommunication:

    def test_shape(self, num_agents, num_messages):
        heuristic = HeuristicCommunication(num_messages=num_messages, grid_size=10)
        state = torch.randn(num_agents, 16)
        agent_pos = torch.randint(0, 10, (num_agents, 2)).float()
        m, logits = heuristic(state, agent_pos=agent_pos)
        assert m.shape == (num_agents, num_messages), \
            f"Expected ({num_agents}, {num_messages}), got {m.shape}"
        assert logits is None

    def test_valid_one_hot(self, num_messages):
        heuristic = HeuristicCommunication(num_messages=num_messages, grid_size=10)
        state = torch.randn(BATCH_SIZE, 16)
        agent_pos = torch.randint(0, 10, (BATCH_SIZE, 2)).float()
        m, _ = heuristic(state, agent_pos=agent_pos)
        assert torch.allclose(m.sum(dim=-1), torch.ones(BATCH_SIZE)), \
            "Heuristic messages must sum to 1."
        assert ((m == 0) | (m == 1)).all(), "Heuristic must produce clean one-hots."

    def test_deterministic(self, num_messages):
        """Same position should always map to the same symbol."""
        heuristic = HeuristicCommunication(num_messages=num_messages, grid_size=10)
        state = torch.randn(BATCH_SIZE, 16)
        agent_pos = torch.randint(0, 10, (BATCH_SIZE, 2)).float()
        m1, _ = heuristic(state, agent_pos=agent_pos)
        m2, _ = heuristic(state, agent_pos=agent_pos)
        assert torch.allclose(m1, m2), "Heuristic must be deterministic."

    def test_symbol_in_range(self, num_messages):
        heuristic = HeuristicCommunication(num_messages=num_messages, grid_size=10)
        state = torch.randn(BATCH_SIZE, 16)
        agent_pos = torch.randint(0, 10, (BATCH_SIZE, 2)).float()
        m, _ = heuristic(state, agent_pos=agent_pos)
        idx = m.argmax(dim=-1)
        assert (idx >= 0).all() and (idx < num_messages).all(), \
            f"Symbol indices must be in [0, {num_messages-1}]."


# ===========================================================================
# 8. Aggregate Helpers
# ===========================================================================

class TestAggregateMessages:

    def test_aggregate_unbatched(self, num_agents, num_messages):
        """aggregate_messages(msg, i) must return (N-1, K) excluding row i."""
        # Build messages so each agent has a UNIQUE row (no two rows identical).
        # We embed agent index as a distinguishable float pattern.
        messages = torch.zeros(num_agents, num_messages)
        for a in range(num_agents):
            messages[a, a % num_messages] = float(a + 1)   # unique non-zero element

        for i in range(num_agents):
            received = aggregate_messages(messages, receiver_id=i)
            assert received.shape == (num_agents - 1, num_messages), \
                f"Expected ({num_agents-1}, {num_messages}), got {received.shape}"
            # Verify agent i's own row is absent from received
            # (compare each received row against messages[i])
            for j in range(num_agents - 1):
                match = torch.allclose(received[j], messages[i])
                # Only fail if the rows are literally identical AND messages[i] != 0
                if messages[i].abs().sum() > 0:
                    assert not match, \
                        f"Agent {i}'s own message should not appear in its received set."

    def test_aggregate_batched(self, num_agents, num_messages):
        B = 4
        messages = torch.randn(B, num_agents, num_messages)
        for i in range(num_agents):
            received = aggregate_messages_batched(messages, receiver_id=i)
            assert received.shape == (B, num_agents - 1, num_messages), \
                f"Expected ({B}, {num_agents-1}, {num_messages}), got {received.shape}"

    def test_zero_messages_shape(self, num_agents, num_messages):
        z = zero_messages(num_agents, num_messages)
        assert z.shape == (num_agents, num_messages)
        assert z.sum().item() == 0.0

    def test_zero_messages_batched_shape(self, num_agents, num_messages):
        B = 4
        z = zero_messages_batched(B, num_agents, num_messages)
        assert z.shape == (B, num_agents, num_messages)
        assert z.sum().item() == 0.0


# ===========================================================================
# 9. Diagnostic Entropy
# ===========================================================================

class TestDiagnosticEntropy:

    def test_entropy_from_logits_shape(self, num_messages):
        logits = torch.randn(BATCH_SIZE, num_messages)
        ent = message_entropy(logits)
        assert ent.shape == (), "entropy must be a scalar"
        assert ent.item() >= 0, "entropy must be non-negative"

    def test_entropy_max_at_uniform(self, num_messages):
        """Uniform logits → max entropy ≈ log(K)."""
        logits = torch.zeros(BATCH_SIZE, num_messages)
        ent = message_entropy(logits)
        expected_max = math.log(num_messages)
        assert abs(ent.item() - expected_max) < 1e-4, \
            f"Uniform logits entropy expected ≈ {expected_max:.4f}, got {ent.item():.4f}"

    def test_entropy_min_at_peaked(self, num_messages):
        """Very peaked logits → entropy ≈ 0."""
        logits = torch.zeros(BATCH_SIZE, num_messages)
        logits[:, 0] = 1e6                    # almost all mass on symbol 0
        ent = message_entropy(logits)
        assert ent.item() < 0.01, f"Peaked logits should have near-zero entropy: {ent.item()}"

    def test_entropy_from_onehots(self, num_messages):
        """Empirical entropy of uniform one-hots ≈ log(K)."""
        B_large = 10000
        torch.manual_seed(0)
        idx = torch.randint(0, num_messages, (B_large,))
        messages = F.one_hot(idx, num_classes=num_messages).float()
        ent = message_entropy_from_onehots(messages)
        expected = math.log(num_messages)
        assert abs(ent.item() - expected) < 0.05, \
            f"Empirical entropy expected ≈ {expected:.4f}, got {ent.item():.4f}"


# ===========================================================================
# 10. CommunicationNetwork Backward-Compat Alias
# ===========================================================================

class TestCommunicationNetworkAlias:

    def test_compat_signature_instantiation(self):
        """Original stub signature must still work."""
        net = CommunicationNetwork(input_dim=32, hidden_dim=128, num_messages=4)
        assert isinstance(net, MessageHead)

    def test_compat_forward_training_true(self):
        net = CommunicationNetwork(input_dim=32, hidden_dim=128, num_messages=4)
        x = torch.randn(BATCH_SIZE, 32)
        m, logits = net(x, training=True)
        assert m.shape == (BATCH_SIZE, 4)
        assert logits.shape == (BATCH_SIZE, 4)

    def test_compat_forward_training_false(self):
        net = CommunicationNetwork(input_dim=32, hidden_dim=128, num_messages=4)
        x = torch.randn(BATCH_SIZE, 32)
        m, logits = net(x, training=False)
        # Eval: same input → same output
        m2, _ = net(x, training=False)
        assert torch.allclose(m, m2), "Compat alias eval must be deterministic."

    def test_compat_message_onehot(self):
        net = CommunicationNetwork(input_dim=32, hidden_dim=128, num_messages=4)
        x = torch.randn(BATCH_SIZE, 32)
        m, _ = net(x, training=False)
        assert torch.allclose(m.sum(dim=-1), torch.ones(BATCH_SIZE))


# ===========================================================================
# 11. AgentMemory Backward-Compat Alias
# ===========================================================================

class TestAgentMemoryAlias:

    def test_compat_instantiation(self):
        mem = AgentMemory(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        assert isinstance(mem, GRUMemory)

    def test_compat_single_step_shape(self):
        mem = AgentMemory(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        out, h = mem(x)
        assert out.shape == (BATCH_SIZE, HIDDEN_DIM)
        assert h.shape == (1, BATCH_SIZE, HIDDEN_DIM)

    def test_compat_sequence_shape(self):
        mem = AgentMemory(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
        L = 7
        x = torch.randn(BATCH_SIZE, L, INPUT_DIM)
        out, h = mem(x)
        assert out.shape == (BATCH_SIZE, L, HIDDEN_DIM)


# ===========================================================================
# 12. MessageEmbedding Gradient Transparency
# ===========================================================================

class TestMessageEmbedding:

    def test_gradient_flows_through_linear_embed(self, num_messages):
        """nn.Linear embedding must pass gradients to upstream one-hot tensors."""
        embed = MessageEmbedding(num_messages=num_messages, embed_dim=16)
        # Simulate a ST-Gumbel one-hot with grad
        logits = torch.randn(BATCH_SIZE, num_messages, requires_grad=True)
        message = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        embedded = embed(message)
        loss = embedded.sum()
        loss.backward()
        assert logits.grad is not None, "Gradients must flow through MessageEmbedding."
        assert logits.grad.abs().max().item() > 0

    def test_embed_shape(self, num_messages):
        embed_dim = 12
        embed = MessageEmbedding(num_messages=num_messages, embed_dim=embed_dim)
        x = torch.randn(BATCH_SIZE, num_messages)
        out = embed(x)
        assert out.shape == (BATCH_SIZE, embed_dim)

    def test_embed_received_batched_flat(self, num_agents, num_messages):
        """embed_received((B, N-1, K)) -> (B, (N-1)*embed_dim) for DQNAgent concat."""
        embed_dim = 8
        embed = MessageEmbedding(num_messages=num_messages, embed_dim=embed_dim)
        B = BATCH_SIZE
        received = torch.randn(B, num_agents - 1, num_messages)
        flat = embed.embed_received(received)
        expected_dim = (num_agents - 1) * embed_dim
        assert flat.shape == (B, expected_dim), \
            f"embed_received batched: expected ({B}, {expected_dim}), got {flat.shape}"

    def test_embed_received_unbatched_flat(self, num_agents, num_messages):
        """embed_received((N-1, K)) -> ((N-1)*embed_dim,) for single-step inference."""
        embed_dim = 8
        embed = MessageEmbedding(num_messages=num_messages, embed_dim=embed_dim)
        received = torch.randn(num_agents - 1, num_messages)
        flat = embed.embed_received(received)
        expected_dim = (num_agents - 1) * embed_dim
        assert flat.shape == (expected_dim,), \
            f"embed_received unbatched: expected ({expected_dim},), got {flat.shape}"

    def test_embed_received_grad_flow(self, num_messages):
        """Gradients must flow through embed_received back to upstream messages."""
        embed = MessageEmbedding(num_messages=num_messages, embed_dim=num_messages)
        logits = torch.randn(BATCH_SIZE, 3, num_messages, requires_grad=True)
        msgs = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        flat = embed.embed_received(msgs)          # (B, 3*embed_dim)
        flat.sum().backward()
        assert logits.grad is not None and logits.grad.abs().max() > 0


# ===========================================================================
# 13.  C6 → MessageEmbedding Pipeline
# ===========================================================================

class TestC6EmbeddingPipeline:
    """
    Verify that RandomCommunication outputs are shape-compatible with
    MessageEmbedding — same (B, K) one-hot format as MessageHead, no gradients.
    """

    def test_c6_into_embedding_shape(self, num_agents, num_messages):
        """Full C6 pipeline: random msg → embed → flat vector."""
        random_comm = RandomCommunication(num_messages=num_messages)
        embed = MessageEmbedding(num_messages=num_messages, embed_dim=num_messages)

        state = torch.randn(num_agents, INPUT_DIM)
        msg, _ = random_comm(state)                        # (N, K)
        embedded = embed(msg)                              # (N, K)  [embed_dim = K]
        assert embedded.shape == (num_agents, num_messages), \
            f"C6 embed shape wrong: {embedded.shape}"

    def test_c6_matches_learned_embed_interface(self, num_messages):
        """MessageHead and RandomCommunication must produce same-shape output."""
        head = MessageHead(input_dim=INPUT_DIM, num_messages=num_messages)
        head.train()
        random_comm = RandomCommunication(num_messages=num_messages)
        state = torch.randn(BATCH_SIZE, INPUT_DIM)

        learned_msg, _ = head(state)
        random_msg, _ = random_comm(state)

        assert learned_msg.shape == random_msg.shape, \
            f"Shape mismatch: learned={learned_msg.shape}, random={random_msg.shape}"

    def test_c6_no_grad_through_embedding(self, num_messages):
        """Embedding C6 messages must not create gradients for comm parameters."""
        random_comm = RandomCommunication(num_messages=num_messages)
        embed = MessageEmbedding(num_messages=num_messages)
        state = torch.randn(BATCH_SIZE, INPUT_DIM, requires_grad=True)
        msg, _ = random_comm(state)
        embedded = embed(msg)
        loss = embedded.sum()
        loss.backward()
        # state.grad is None because msg was detached — gradient stops at C6 output
        assert state.grad is None, \
            "No gradient should propagate back through C6 messages to state."

    def test_c6_embed_received_pipeline(self, num_agents, num_messages):
        """Full pipeline: build msg bank from C6, aggregate, embed_received, flatten."""
        random_comm = RandomCommunication(num_messages=num_messages)
        embed = MessageEmbedding(num_messages=num_messages, embed_dim=8)
        state = torch.randn(num_agents, INPUT_DIM)

        # Collect per-agent messages into a bank (N, K)
        msgs_list = []
        for i in range(num_agents):
            m, _ = random_comm(state[i:i+1])    # (1, K)
            msgs_list.append(m)
        msg_bank = torch.cat(msgs_list, dim=0)  # (N, K)

        # Agent 0 aggregates and embeds
        received = aggregate_messages(msg_bank, receiver_id=0)   # (N-1, K)
        flat = embed.embed_received(received)                     # ((N-1)*8,)
        assert flat.shape == ((num_agents - 1) * 8,), \
            f"C6 embed_received flat shape wrong: {flat.shape}"


# ===========================================================================
# 14.  Full C1–C6 DQNAgent & ReplayBuffer Integration
# ===========================================================================

class TestC1ThroughC6Integration:
    """
    Verify that DQNAgent and ReplayBuffer instantiate, select actions, and
    train across all 6 experimental conditions without shape or gradient errors.
    """

    @pytest.mark.parametrize("condition", ["C1", "C2", "C3", "C4", "C5", "C6"])
    def test_agent_action_selection(self, condition):
        from agents.dqn import DQNAgent
        from training.config import get_config
        import numpy as np

        cfg = get_config(condition)
        obs_dim = 302 if cfg["observation_radius"] == "full" else 77
        N = cfg["num_agents"]

        agent = DQNAgent(
            input_dim=obs_dim,
            num_agents=N,
            num_actions=5,
            hidden_dim=32,
            use_memory=cfg["use_memory"],
            comm_mode=cfg["comm_mode"],
            num_messages=cfg["num_messages"],
        )

        obs = np.random.randn(N, obs_dim).astype(np.float32)
        actions = agent.select_actions(obs, epsilon=0.0)

        assert actions.shape == (N,)
        assert (actions >= 0).all() and (actions < 5).all()
        if cfg["comm_mode"] != "none":
            assert agent.current_messages is not None
            assert agent.current_messages.shape == (N, cfg["num_messages"])

    @pytest.mark.parametrize(
        "condition, per_agent_rewards",
        [
            ("C1", False),
            ("C2", False),
            ("C3", False),
            ("C3", True),
            ("C4", False),
            ("C5", False),
            ("C5", True),
            ("C6", False),
        ],
    )
    def test_agent_training_step(self, condition, per_agent_rewards):
        from agents.dqn import DQNAgent
        from agents.replay_buffer import ReplayBuffer
        from training.config import get_config
        import numpy as np

        cfg = get_config(condition)
        obs_dim = 302 if cfg["observation_radius"] == "full" else 77
        N = cfg["num_agents"]
        K = cfg["num_messages"]

        agent = DQNAgent(
            input_dim=obs_dim,
            num_agents=N,
            num_actions=5,
            hidden_dim=32,
            batch_size=4,
            use_memory=cfg["use_memory"],
            comm_mode=cfg["comm_mode"],
            num_messages=K,
        )
        buffer = ReplayBuffer(capacity=100)
        eye = np.eye(N, dtype=np.float32)

        # Populate buffer with dummy transitions / episodes
        for ep in range(6):
            ep_trans = []
            agent.reset_hidden()
            prev_m = agent.init_messages()
            for t in range(5):
                obs = np.random.randn(N, obs_dim).astype(np.float32)
                act = agent.select_actions(obs, epsilon=0.1, prev_messages=prev_m)
                cur_m = agent.current_messages
                next_obs = np.random.randn(N, obs_dim).astype(np.float32)
                reward = np.ones(N, dtype=np.float32) if per_agent_rewards else 1.0
                trans = (obs, eye, act, reward, next_obs, False, prev_m, cur_m)
                buffer.add(trans)
                ep_trans.append(trans)
                prev_m = cur_m
            buffer.add_episode(ep_trans)

        loss = agent.train_step(buffer)
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0.0
