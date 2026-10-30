import torch
import torch.nn as nn
import torch.nn.functional as F

class CommunicationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_messages=4):
        super(CommunicationNetwork, self).__init__()
        # Build a simple feedforward neural network with two hidden layers using nn.Linear and ReLU activation functions.
        # that predicts communication actions given the input state.
        self.num_messages = num_messages
        #TODO: this network generate message scores.
        self.message_head=nn.Linear(hidden_dim, num_messages)

    def forward(self,encoded_observation,training=True):
        """
        TODO: 
        1. Generate message scores from the input observations using the network.
        2. Use straight-through Gumbel-Softmax to sample a message from the message scores.
        3. use argmax to get the index of the sampled message.
        4. return the sampled message and the index of the sampled message.
        """
        logits = self.message_head(encoded_observation)
        # Sample a message using Gumbel-Softmax
        if training:
            message = F.gumbel_softmax(
                logits, tau=1.0, hard=True, dim=-1
            )
        else:
            message_id = logits.argmax(dim=-1)
            message = F.one_hot(
                message_id, num_classes=self.num_messages
            ).float()
        return message, logits
    
def aggregate_messages(messages,receiver_id):
    """
    TODO:
    Aggregate messages from multiple agents for a specific receiver agent.
    messages: A tensor of shape (num_agents, num_messages) containing the messages from all agents.
    """
    
    raise NotImplementedError("aggregate_messages function not implemented yet.")