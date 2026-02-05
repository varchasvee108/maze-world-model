import numpy as np
import random


class MazeTransitionGenerator:
    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }
    LABELS = {
        "STAY": 0,
        "UP": 1,
        "DOWN": 2,
        "LEFT": 3,
        "RIGHT": 4,
    }

    def __init__(self, grid_size=10):
        self.grid_size = grid_size

    def generate_solvable_maze(self):
        N = self.grid_size
        maze = np.ones((N, N), dtype=np.uint8)

        def carve(x, y):
            maze[x, y] = 0
            directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < N and maze[nx, ny] == 1:
                    maze[x + dx // 2, y + dy // 2] = 0
                    carve(nx, ny)

        carve(0, 0)
        maze[0, 0] = 0
        maze[N - 1, N - 1] = 0
        maze[N - 2, N - 1] = 0
        maze[N - 1, N - 2] = 0
        return maze

    def compute_physics(self, maze, pos, action):
        dx, dy = self.ACTIONS[action]
        x, y = pos
        nx, ny = x + dx, y + dy

        if (
            nx < 0
            or nx >= self.grid_size
            or ny < 0
            or ny >= self.grid_size
            or maze[nx, ny] == 1
        ):
            return pos, self.LABELS["STAY"]

        if action == 0:
            return (nx, ny), self.LABELS["UP"]
        elif action == 1:
            return (nx, ny), self.LABELS["DOWN"]
        elif action == 2:
            return (nx, ny), self.LABELS["LEFT"]
        elif action == 3:
            return (nx, ny), self.LABELS["RIGHT"]

    def get_maze_transitions(self):

        maze = self.generate_solvable_maze()
        free_cells = np.argwhere(maze == 0)

        exit_pos = (self.grid_size - 1, self.grid_size - 1)

        transitions = []

        for x, y in free_cells:
            for action in range(4):
                next_pos, label = self.compute_physics(maze, (x, y), action)
                transitions.append(
                    {
                        "maze": maze.copy(),
                        "start": (x, y),
                        "action": action,
                        "label": label,
                        "exit": exit_pos,
                    }
                )
        return transitions
