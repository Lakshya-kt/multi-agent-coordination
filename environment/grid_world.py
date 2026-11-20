import numpy as np
from environment.observations import  get_all_observations
# action index -> (d_row, d_col): 0 up, 1 down, 2 left, 3 right, 4 stay
MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)])

class GridWorld:
    
    def __init__(self, grid_size=10, num_agents=4, num_targets=4, max_steps=50,
                 radius=2, seed=None, observation_radius=None):
        if observation_radius is not None:
            radius = observation_radius
        # Spec: |targets| = N, T = 50, N = 4 in the main grid.
        assert num_targets == num_agents, "spec requires |Gamma| = N"
        assert num_targets <= grid_size ** 2
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.max_steps = max_steps
        self.radius = radius                    # int >= 1, or "full"
        self.full = (radius == "full")
        self.rng = np.random.default_rng(seed)
        self.n_actions = len(MOVES)
        w = grid_size if self.full else 2 * radius + 1
        self.obs_shape = (3, w, w)              # channels: targets, others, walls
        self.agent_position = None              # (N, 2) int array of (row, col)
        self.target_position = None             # (N, 2) int array, distinct
        self.step_count = 0
    
    def get_obs(self):
        return get_all_observations(self.agent_position, self.target_position,
                                self.grid_size, self.radius)
    
    def reset(self,seed=None):
        """
        TODO:
        1. Randomly initialize the positions of agents and targets in the grid.
        2. Ensure that no two agents or targets occupy the same position.
        3. Reset the step count to zero.
        4. Return the initial state of the environment, which includes the positions of agents and targets.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        G=self.grid_size
        #TODO: Randomly initialize the positions of agents and targets in the grid
        cells = self.rng.choice(G*G, size=self.num_targets, replace=False)
        self.target_position = np.stack([cells // G, cells % G], axis=1) 
        while True:
            self.agent_position = self.rng.integers(0, G, size=(self.num_agents, 2))
           
            if not self.all_covered():
                break
        self.step_count = 0
        return self.get_obs()  # Replace with the initial state of the environment
    
    def step(self, actions):
        """
        TODO:
        1. Apply action for each agent to update their positions in the grid.
        2. Ensure that agents do not move outside the grid boundaries.
        3. Check if any agent has reached a target and update the state accordingly.
        4. Increment the step count.
        5. Return the new state of the environment, a reward signal, and a boolean indicating if the episode is done.
        """
        before=self.get_covered_targets()
        
        new_positions = self.agent_position + MOVES[np.asarray(actions)]
        inside=((new_positions >= 0) & (new_positions < self.grid_size)).all(axis=1)
        self.agent_position = np.where(inside[:, None], new_positions, self.agent_position)
        
        #increment step count
        self.step_count += 1
        
        #coverage after the move : C(s_t+1)
        after=self.get_covered_targets()
        
        #reward is the number of new targets covered by agents in this step
        reward=max(after-before,0) -0.01 #penalty for each step to encourage faster coverage
        if after==self.num_targets:
            reward+=5 #bonus for covering all targets
        
        ## episode end
        terminated=(after==self.num_targets)
        truncated=(not terminated) and self.step_count>=self.max_steps
        info = {"covered": after, "success": bool(terminated),
        "truncated": bool(truncated), "t": self.step_count}
        return self.get_obs(), reward, terminated or truncated, info
    
    
    def get_covered_targets(self):
        """
        TODO: Return the number of targets that have been covered by agents.
        """
        occupied = set(tuple(pos) for pos in self.agent_position)
        covered = sum(tuple(target) in occupied for target in self.target_position)
        return covered
        
    def all_covered(self):
        return self.get_covered_targets() == self.num_targets