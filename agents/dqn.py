import torch
import torch.nn as nn
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128,num_actions=5):
        super(DQN, self).__init__()
        # Todo:
        # Build a simple feedforward neural network with two hidden layers using nn.Linear and ReLU activation functions.
        # that predicts Q-values for each action given the input state.
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, observations):
        return self.network(observations)

class DQNAgent:
    def __init__(self, input_dim, num_actions=5):
        self.num_actions = num_actions
        self.policy_net = DQN(input_dim, num_actions=num_actions)
        self.target_net = DQN(input_dim, num_actions=num_actions)
        
        #TODO:
        # Initialize the target network with the same weights as the policy network
        #set up the optimizer (e.g., Adam),epsilon,gamma and learning rate for training the policy network
        
    def select_actions(self,observations, epsilon=0.1):
        """
        TODO: Implement epsilon-greedy action selection
        """
        raise NotImplementedError("select_actions method not implemented yet.")
    
    def train_step(self, replay_buffer):
        """
        TODO:
        Implement a single training step for the DQN agent
        1. Sample a batch of experiences from the replay buffer
        2.calculate Q-values for the current states using the policy network
        3.calculate target Q-values using the target network and the Bellman equation
        4.compute the loss between the predicted Q-values and the target Q-values
        5.update the policy network using backpropagation and the optimizer
        """
        raise NotImplementedError("train_step method not implemented yet.")
    
    def update_target_network(self):
        """
        TODO: Copy the weights from the policy network to the target network
        """
        self.target_network.load_state_dict(self.policy_net.state_dict())
        