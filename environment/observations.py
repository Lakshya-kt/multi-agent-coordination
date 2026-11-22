import numpy as np

def _maps(agent_position, target_position, grid_size):
    G = grid_size
    targets = np.zeros((G, G), np.float32)
    targets[target_position[:, 0], target_position[:, 1]] = 1.0
    counts = np.zeros((G, G), np.int32)
    np.add.at(counts, (agent_position[:, 0], agent_position[:, 1]), 1)
    return targets, counts


def get_local_observations(agent_id, agent_position, target_position, grid_size, radius):
    """
    TODO: Create a local observation for the specified agent based on its position and the given radius.
    
    Includes:
    1. Target channels: A binary representation of the presence of targets within the specified radius around the agent.
    2. Agent channels: A binary representation of the presence of other agents within the specified radius around the agent.
    3. The local observation should be a 3D array with shape (2, 2*radius+1, 2*radius+1), where the first channel corresponds to targets and the second channel corresponds to agents.
    4. Ensure that the local observation does not exceed the grid boundaries.
    """
    G=grid_size
    
    targets,counts=_maps(agent_position,target_position,G)
    y,x=agent_position[agent_id]
    counts[y,x]-=1  # don't count self
    others=(counts>0).astype(np.float32)
    walls=np.zeros((G,G),np.float32)
    
    if radius=="full":
        self_channel = np.zeros((G, G), np.float32)
        self_channel[y, x] = 1.0
        return np.stack([targets, others, self_channel])

    ## pad so a window centred near an edge never goes out of bounds
    r=radius
    T = np.pad(targets, r)                                 # outside grid: no target
    A = np.pad(others, r)                                  # outside grid: no agent
    W = np.pad(walls, r, constant_values=1)
 
    sl=slice(y, y+2*r+1), slice(x, x+2*r+1)
    return np.stack([T[sl], A[sl], W[sl]])
    

def get_all_observations(agent_position, target_position, grid_size,radius):
    """Full-grid view for every agent (condition C1). Shape (N, 3, G, G)."""
    n=len(agent_position)
    windows=np.stack([get_local_observations(i, agent_position, target_position, grid_size, radius) for i in range(n)])
    pbar = (agent_position / (grid_size - 1)).astype(np.float32)
    return windows, pbar


