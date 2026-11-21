"""
evaluation/baselines.py — Decentralized non-learning greedy heuristic baseline.

Authoritative reference: Group_9_FAI_Project_Idea.pdf Section 4:
  "A non-learning greedy heuristic baseline (move to the nearest visible unassigned
   target, else random-walk)."

Constraints:
  - Uses ONLY local observation window channels available to agent i under radius r.
  - Channel 0: Visible targets
  - Channel 1: Other visible agents
  - Channel 2: Wall padding
  - Never accesses global target or teammate positions unavailable to a local agent.
  - Action interface: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay (matches grid_world.py).
"""

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np


# Action mapping matching environment/grid_world.py MOVES:
# 0: (-1, 0) Up
# 1: ( 1, 0) Down
# 2: ( 0, -1) Left
# 3: ( 0,  1) Right
# 4: ( 0,  0) Stay
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]


class GreedyHeuristicPolicy:
    """
    Decentralized greedy heuristic policy for multi-agent target coverage.

    Policy logic per agent i:
      1. Inspect local observation window of shape (3, W, W).
         Center is at (r, r) for local radius r, or agent's (row, col) for full grid.
      2. If agent is currently on an unoccupied target -> action 4 (Stay).
      3. Find all visible targets (channel 0 == 1).
      4. Filter out targets currently occupied by other visible agents (channel 1 == 1).
      5. If unassigned visible targets exist:
         - Select the nearest one by Manhattan distance |dy| + |dx|.
         - Move one step closer along the axis with the larger displacement.
      6. If no unassigned targets visible:
         - Perform random walk among valid moves that do not collide with walls (channel 2 == 0).
    """

    def __init__(self, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None):
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def select_agent_action(
        self,
        window: np.ndarray,
        pbar: Optional[np.ndarray] = None,
        radius: Union[int, str] = 2,
    ) -> int:
        """
        Select action for a single agent from its (3, W, W) observation window.

        Args:
            window: (3, W, W) array.
                    Channel 0: targets
                    Channel 1: other agents
                    Channel 2: walls (1 = wall/out-of-bounds, 0 = open)
            pbar: Optional (2,) normalized position [row/9, col/9]
            radius: int or "full"

        Returns:
            int action in {0, 1, 2, 3, 4}
        """
        _, H, W = window.shape
        targets_chan = window[0]
        others_chan = window[1]
        walls_chan = window[2]

        is_full = str(radius) == "full"
        if is_full:
            # Under full observation, agent's own position is derived from pbar
            if pbar is not None:
                center_y = int(round(pbar[0] * (H - 1)))
                center_x = int(round(pbar[1] * (W - 1)))
            else:
                # Fallback: full grid without self-identifier
                center_y, center_x = H // 2, W // 2
        else:
            center_y, center_x = H // 2, W // 2

        # Check if agent is currently on a target
        on_target = bool(targets_chan[center_y, center_x] > 0.5)
        # If already on a target and no other agent is here, stay!
        if on_target:
            return 4  # Stay

        # Locate all visible target coordinates in the window
        target_indices = np.argwhere(targets_chan > 0.5)
        if len(target_indices) == 0:
            return self._random_walk(walls_chan, center_y, center_x)

        # Separate targets into unoccupied vs occupied by other visible agents
        unoccupied_targets = []
        occupied_targets = []
        for ty, tx in target_indices:
            if others_chan[ty, tx] > 0.5:
                occupied_targets.append((ty, tx))
            else:
                unoccupied_targets.append((ty, tx))

        # Prefer unoccupied targets; fall back to any visible target if all occupied
        candidate_targets = unoccupied_targets if unoccupied_targets else occupied_targets

        # Find nearest target by Manhattan distance from agent center
        best_target = None
        min_dist = float("inf")
        for ty, tx in candidate_targets:
            dist = abs(ty - center_y) + abs(tx - center_x)
            if dist < min_dist:
                min_dist = dist
                best_target = (ty, tx)

        if best_target is None:
            return self._random_walk(walls_chan, center_y, center_x)

        ty, tx = best_target
        dy = ty - center_y
        dx = tx - center_x

        # Determine directional step towards target
        return self._step_towards(dy, dx, walls_chan, center_y, center_x)

    def select_actions(
        self,
        raw_obs: Tuple[np.ndarray, np.ndarray],
        radius: Union[int, str] = 2,
    ) -> np.ndarray:
        """
        Select actions for all N agents.

        Args:
            raw_obs: Tuple (windows, pbar) as returned by GridWorld.
                     windows shape: (N, 3, W, W)
                     pbar shape: (N, 2)
            radius: Observation radius

        Returns:
            np.ndarray of shape (N,) with integer actions in {0, 1, 2, 3, 4}.
        """
        windows, pbar = raw_obs
        num_agents = len(windows)
        actions = np.zeros(num_agents, dtype=np.int64)

        for i in range(num_agents):
            pbar_i = pbar[i] if pbar is not None else None
            actions[i] = self.select_agent_action(
                window=windows[i],
                pbar=pbar_i,
                radius=radius,
            )

        return actions

    def _step_towards(
        self,
        dy: int,
        dx: int,
        walls: np.ndarray,
        cy: int,
        cx: int,
    ) -> int:
        """Choose best valid action reducing (dy, dx) without hitting walls."""
        candidates = []
        if abs(dy) >= abs(dx):
            # Prefer vertical
            primary_act = 0 if dy < 0 else 1
            secondary_act = (2 if dx < 0 else 3) if dx != 0 else None
        else:
            # Prefer horizontal
            primary_act = 2 if dx < 0 else 3
            secondary_act = (0 if dy < 0 else 1) if dy != 0 else None

        candidates.append(primary_act)
        if secondary_act is not None:
            candidates.append(secondary_act)

        # Check validity (not stepping into wall)
        H, W = walls.shape
        for act in candidates:
            step_y, step_x = ACTION_DELTAS[act]
            ny, nx = cy + step_y, cx + step_x
            if 0 <= ny < H and 0 <= nx < W and walls[ny, nx] < 0.5:
                return act

        # If ideal steps are blocked by walls, try any open movement
        return self._random_walk(walls, cy, cx)

    def _random_walk(self, walls: np.ndarray, cy: int, cx: int) -> int:
        """Select a random open action that does not move into a wall."""
        H, W = walls.shape
        valid_actions = []
        for act in range(4):  # Exclude 4 (stay) from exploration walk
            step_y, step_x = ACTION_DELTAS[act]
            ny, nx = cy + step_y, cx + step_x
            if 0 <= ny < H and 0 <= nx < W and walls[ny, nx] < 0.5:
                valid_actions.append(act)

        if not valid_actions:
            return 4  # Stay if all directions blocked
        return int(self.rng.choice(valid_actions))
