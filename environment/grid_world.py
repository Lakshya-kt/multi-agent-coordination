import numpy as np

class GridWorld:
    
    def __init__(self, grid_size=10, num_agents=2, num_targets=2,max_steps=100):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.max_steps = max_steps
        self.agent_position= None
        self.target_position = None
        self.step_count = 0
    
    def reset(self,seed=None):
        """
        TODO:
        1. Randomly initialize the positions of agents and targets in the grid.
        2. Ensure that no two agents or targets occupy the same position.
        3. Reset the step count to zero.
        4. Return the initial state of the environment, which includes the positions of agents and targets.
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.step_count = 0
        #TODO: Randomly initialize the positions of agents and targets in the grid
        return None  # Replace with the initial state of the environment
    
    def step(self, actions):
        """
        TODO:
        1. Apply action for each agent to update their positions in the grid.
        2. Ensure that agents do not move outside the grid boundaries.
        3. Check if any agent has reached a target and update the state accordingly.
        4. Increment the step count.
        5. Return the new state of the environment, a reward signal, and a boolean indicating if the episode is done.
        """
        self.step_count += 1
        raise NotImplementedError("TODO: The step function is not implemented yet.")
        
    def get_covered_targets(self):
        """
        TODO: Return the number of targets that have been covered by agents.
        """
        raise NotImplementedError("TODO: The get_covered_targets function is not implemented yet.")