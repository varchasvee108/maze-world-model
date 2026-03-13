import torch
import torch.nn as nn
from core.config import Config


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
        self.row_embd = nn.Parameter(torch.randn(grid_size, n_embd) * 0.02)
        self.col_embd = nn.Parameter(torch.randn(grid_size, n_embd) * 0.02)
        self.cls_token_embd = nn.Parameter(torch.randn(1, 1, n_embd) * 0.02)

    def forward(self, x):
        B, T, C = x.shape
        row_embd = self.row_embd.unsqueeze(1)
        col_embd = self.col_embd.unsqueeze(0)

        pos_embd = row_embd + col_embd
        pos_embd = pos_embd.view(1, -1, C)
        cls_embd = self.cls_token_embd
        final_pos_embd = torch.cat([cls_embd, pos_embd], dim=1)
        return x + final_pos_embd


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.model.n_embd)
        self.ln_2 = nn.LayerNorm(config.model.n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.model.n_embd,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.model.n_embd, 4 * config.model.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.model.n_embd, config.model.n_embd),
            nn.Dropout(config.model.dropout),
        )

    def forward(self, x, return_attn=False):
        norm_x = self.ln_1(x)

        attn_out, weights = self.attn(
            norm_x,
            norm_x,
            norm_x,
            need_weights=return_attn,
            average_attn_weights=True,
        )

        x = x + attn_out

        x = x + self.mlp(self.ln_2(x))

        return (x, weights) if return_attn else x
