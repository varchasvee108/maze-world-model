from torch.utils.data import DataLoader, Subset
from maze_dataset.dataset import MazeVisionDataset
import lightning as L
import numpy as np
from pathlib import Path


class MazeDataModule(L.LightningDataModule):
    def __init__(self, npz_path, batch_size=64, train_split=0.9):
        super().__init__()
        self.npz_path = Path(npz_path)
        self.batch_size = batch_size
        self.train_split = train_split

    def setup(self, stage=None):
        full_dataset = MazeVisionDataset(self.npz_path, img_size=(132, 132))

        maze_indices_per_sample = full_dataset.samples[:, 0]
        unique_maze_indices = np.unique(maze_indices_per_sample)

        rng = np.random.default_rng(42)
        rng.shuffle(unique_maze_indices)

        split_idx = int(len(unique_maze_indices) * self.train_split)

        train_maze_ids = set(unique_maze_indices[:split_idx])
        val_maze_ids = set(unique_maze_indices[split_idx:])

        train_indices = []
        val_indices = []

        for i, maze_id in enumerate(maze_indices_per_sample):
            if maze_id in train_maze_ids:
                train_indices.append(i)
            else:
                val_indices.append(i)

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
