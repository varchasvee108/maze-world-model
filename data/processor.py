import numpy as np
import os
from tqdm import tqdm
from data.generator import MazeTransitionGenerator


def generate_full_dataset(
    num_mazes=100000, grid_size=11, save_path="data/maze_data.npz"
):
    generator = MazeTransitionGenerator(grid_size=grid_size)
    maze_library = []
    maze_exits = []
    maze_info = []

    print(f"Constructing {num_mazes} mazes...")

    for maze_idx in tqdm(range(num_mazes)):
        maze_samples = generator.generate_policy_samples()

        if not maze_samples:
            continue
        maze_library.append(maze_samples[0]["maze"])
        maze_exits.append(maze_samples[0]["exit"])

        for s in maze_samples:
            maze_info.append(
                [
                    maze_idx,
                    s["start"][0],
                    s["start"][1],
                    s["label"],
                ]
            )

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        save_path,
        mazes=np.array(maze_library, dtype=np.uint8),
        exits=np.array(maze_exits, dtype=np.uint16),
        samples=np.array(maze_info, dtype=np.uint32),
    )

    print(f"\n Mazes generated successfully! Saved {len(maze_info)} total samples")


if __name__ == "__main__":
    generate_full_dataset()
