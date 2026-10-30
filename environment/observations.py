def get_local_observations(self, agent_id,radius=2):
    """
    TODO: Create a local observation for the specified agent based on its position and the given radius.
    
    Includes:
    1. Target channels: A binary representation of the presence of targets within the specified radius around the agent.
    2. Agent channels: A binary representation of the presence of other agents within the specified radius around the agent.
    3. The local observation should be a 3D array with shape (2, 2*radius+1, 2*radius+1), where the first channel corresponds to targets and the second channel corresponds to agents.
    4. Ensure that the local observation does not exceed the grid boundaries.
    """
    raise NotImplementedError("TODO: The get_local_observations function is not implemented yet.")

def get_global_observations(self):
    """
    TODO: Create a global observation for the entire grid world.
    """
    observations=[]
    for agent_id in range(self.num_agents):
        local_obs = self.get_local_observations(agent_id)
        observations.append(local_obs)
        
    return observations
