import random
from collections import deque

class ReplayBuffer:
    def __init__(self,capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self,experience):
        """
        TODO: Add experience to the buffer
        Store one experience in the buffer. If the buffer is full, the oldest experience will be removed to make room for the new one.
        (state, action, reward, next_state, done)
        """
        
        self.buffer.append(experience)
    
    def sample(self,batch_size):
        """
        TODO: Sample a batch of experiences from the buffer
        """
        return random.sample(self.buffer, batch_size)
    
    def len(self):
        return len(self.buffer)
        