import torch
import torch.nn as nn

class AgentMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AgentMemory, self).__init__()
        # Build a simple feedforward neural network with two hidden layers using nn.Linear and ReLU activation functions.
        # that predicts the next state given the current state and action.
        # TODO: Implement a GRU-based memory module that takes in the current observation and outputs a hidden state representing the agent's memory.
        self.gru=nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, observation_features, hidden_state=None):
        """
        TODO: Process the current observation features through the GRU to update the agent's memory.
        observation_features: A tensor of shape (batch_size, input_dim) representing the current observation features.
        """
        
        output, hidden_state = self.gru(observation_features.unsqueeze(1), hidden_state)
        return output, hidden_state
    