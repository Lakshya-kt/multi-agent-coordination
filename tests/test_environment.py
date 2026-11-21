import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from environment.grid_world import GridWorld

def test_environment_reset():
    # TODO: Check that reset initializes agents and targets.
    env = GridWorld(radius=2)
    wins, pbar = env.reset(seed=0)
    # positions are initialized with the right shapes and inside the grid
    assert env.agent_position.shape == (env.num_agents, 2)
    assert env.target_position.shape == (env.num_targets, 2)
    assert ((env.agent_position >= 0) & (env.agent_position < env.grid_size)).all()
    assert ((env.target_position >= 0) & (env.target_position < env.grid_size)).all()

    # targets are distinct (agents may share cells, per the spec)
    assert len({tuple(t) for t in env.target_position}) == env.num_targets

    # step count reset, episode not already solved
    assert env.step_count == 0
    assert not env.all_covered()

    # observation shapes
    assert wins.shape == (env.num_agents, *env.obs_shape)
    assert pbar.shape == (env.num_agents, 2)

    # same seed -> identical episode
    a1, t1 = env.agent_position.copy(), env.target_position.copy()
    env.reset(seed=0)
    assert (env.agent_position == a1).all() and (env.target_position == t1).all()

def test_agent_boundaries():
    # TODO: Check that agents cannot leave the grid.
    env = GridWorld(radius=1)
    env.reset(seed=0)
    G = env.grid_size

    # top-left corner: up (0) and left (2) must be no-ops
    env.agent_position[:] = [[0, 0]] * env.num_agents
    env.step([0, 2, 0, 2])
    assert (env.agent_position == 0).all()

    # bottom-right corner: down (1) and right (3) must be no-ops
    env.agent_position[:] = [[G - 1, G - 1]] * env.num_agents
    env.step([1, 3, 1, 3])
    assert (env.agent_position == G - 1).all()

    # a valid move still works (right from (0,0) -> (0,1))
    env.agent_position[:] = [[0, 0]] * env.num_agents
    env.step([3, 4, 4, 4])
    assert tuple(env.agent_position[0]) == (0, 1)


def test_target_coverage():
    # TODO: Check that each target is counted only once.
    env = GridWorld(num_agents=3, num_targets=3, radius=1)
    env.reset(seed=0)
    env.target_position[:] = [[2, 2], [5, 5], [8, 8]]

    # two agents on one target, one agent on another: 2 covered, not 3
    env.agent_position[:] = [[2, 2], [2, 2], [5, 5]]
    assert env.get_covered_targets() == 2

    # no agents on targets
    env.agent_position[:] = [[0, 0], [0, 1], [0, 2]]
    assert env.get_covered_targets() == 0

    # all targets covered, one agent each
    env.agent_position[:] = [[2, 2], [5, 5], [8, 8]]
    assert env.get_covered_targets() == 3 and env.all_covered()

def oracle_actions(env):
    """Agent i walks to target i using the true state (test only): rows first, then columns."""
    acts = []
    for i in range(env.num_agents):
        dy, dx = env.target_position[i] - env.agent_position[i]
        if dy < 0:   acts.append(0)   # up
        elif dy > 0: acts.append(1)   # down
        elif dx < 0: acts.append(2)   # left
        elif dx > 0: acts.append(3)   # right
        else:        acts.append(4)   # stay
    return acts


def test_oracle_solves_task():
    for n in [3, 4, 6]:
        for r in [1, 2, 3, 5, "full"]:
            env = GridWorld(num_agents=n, num_targets=n, radius=r)
            for ep in range(50):
                env.reset(seed=ep)
                done = False
                while not done:
                    _, _, done, info = env.step(oracle_actions(env))
                assert info["success"], (n, r, ep)
                assert info["t"] <= 18          # max Manhattan distance on a 10x10 grid


def test_truncation_at_horizon():
    env = GridWorld(radius=1)
    env.reset(seed=0)
    env.target_position[:] = [[0, 0], [0, 1], [0, 2], [0, 3]]
    env.agent_position[:] = [[9, 9]] * 4
    for t in range(1, env.max_steps):
        _, _, done, info = env.step([4] * 4)
        assert not done and not info["truncated"], t     # not before step 50
    _, _, done, info = env.step([4] * 4)
    assert done and info["truncated"] and not info["success"]


def test_completion_reward():
    env = GridWorld(num_agents=2, num_targets=2, radius=1)
    env.reset(seed=1)
    env.target_position[:] = [[3, 3], [3, 5]]
    env.agent_position[:] = [[3, 2], [3, 5]]             # agent 1 already covers a target
    _, rew, done, info = env.step([3, 4])                # agent 0 steps onto the other
    assert done and info["success"] and abs(rew - 5.99) < 1e-9


def test_leaving_a_covered_target_is_penalized():
    env = GridWorld(num_agents=2, num_targets=2, radius=1)
    env.reset(seed=1)
    env.target_position[:] = [[3, 3], [3, 5]]
    env.agent_position[:] = [[3, 3], [0, 0]]

    _, reward, done, info = env.step([3, 4])

    assert reward == -1.01
    assert not done
    assert info["covered"] == 0