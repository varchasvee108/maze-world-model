import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data.renderer import MazeRenderer
from pathlib import Path


class MazeVisionDataset(Dataset):
    def __init__(self, npz_path: str, img_size=(132, 132)):
        path = Path(npz_path)
        data = np.load(path, mmap_mode="r")
        self.mazes = data["mazes"]
        self.exits = data["exits"]
        self.samples = data["samples"]

        self.renderer = MazeRenderer(grid_size=self.mazes.shape[1], image_size=img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        maze_idx, x, y, label = self.samples[idx]
        maze = self.mazes[maze_idx]
        exit_pos = self.exits[maze_idx]
        start_pos = (int(x), int(y))

        image = self.renderer.render(maze=maze, agent_pos=start_pos, exit_pos=exit_pos)
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
