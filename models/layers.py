import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=132, patch_size=12, in_channels=3, embd_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embd_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, n_embd, grid_size=11):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = num_patches
        self.n_embd = n_embd
        self.row_embd = nn.Paramter(torch.randn(grid_size, n_embd))
        self.col_embd = nn.Paramter(torch.randn(grid_size, n_embd))

    def forward(self, x):
        B, T, C = x.shape
        row_embd = self.row_embd.unsqueeze(1)
        col_embd = self.col_embd.unsqueeze(0)

        pos_embd = row_embd + col_embd
        pos_embd = pos_embd.view(-1, C)
        return x + pos_embd.unsqueeze(0)
