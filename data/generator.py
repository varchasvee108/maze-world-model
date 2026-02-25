import numpy as np
import random
from collections import deque


class MazeTransitionGenerator:
    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, grid_size=11):
        assert grid_size % 2 == 1
        self.grid_size = grid_size

    def generate_solvable_maze(self):
        N = self.grid_size
        maze = np.ones((N, N), dtype=np.uint8)

        def carve(x, y):
            maze[x, y] = 0
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < N and maze[nx, ny] == 1:
                    maze[x + dx // 2, y + dy // 2] = 0
                    carve(nx, ny)

        carve(0, 0)
        return maze

    def _get_valid_path(self, maze, exit_pos):
        N = self.grid_size
        queue = deque([exit_pos])
        visited = {exit_pos}

        policy = {}

        while queue:
            x, y = queue.popleft()

            for action, (dx, dy) in self.ACTIONS.items():
                nx, ny = x + dx, y + dy

                if (
                    0 <= nx < N
                    and 0 <= ny < N
                    and maze[nx, ny] == 0
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))

                    reverse_action = self._reverse_action(action)
                    policy[(nx, ny)] = reverse_action

        return policy

    def _reverse_action(self, action):
        return {0: 1, 1: 0, 2: 3, 3: 2}[action]

    def generate_policy_samples(self):
        maze = self.generate_solvable_maze()
        free_cells = np.argwhere(maze == 0)
        exit_idx = np.random.choice(len(free_cells))
        exit_pos = tuple(free_cells[exit_idx])

        policy = self._get_valid_path(maze, exit_pos)

        samples = []

        for x, y in free_cells:
            start_pos = (x, y)

            if start_pos == exit_pos:
                continue

            if start_pos not in policy:
                continue

            samples.append(
                {
                    "maze": maze,
                    "start": start_pos,
                    "exit": exit_pos,
                    "label": policy[start_pos],
                }
            )

        return samples
