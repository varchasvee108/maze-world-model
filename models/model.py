import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from models.layers import PatchEmbedding, PositionEmbedding, TransformerBlock
from transformers import get_scheduler
from core.config import Config


class MazeTransformer(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.patch_embd = PatchEmbedding(
            embd_dim=config.model.n_embd,
            img_size=config.input_data.image_size,
            patch_size=config.model.patch_size,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.model.n_embd) * 0.02)

        grid_size = config.input_data.image_size[0] // config.model.patch_size

        self.pos_embd = PositionEmbedding(
            num_patches=grid_size * grid_size,
            n_embd=config.model.n_embd,
            grid_size=grid_size,
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(config=config) for _ in range(config.model.num_layers)]
        )

        self.ln_f = nn.LayerNorm(config.model.n_embd)
        self.head = nn.Linear(config.model.n_embd, 4)

    def forward(self, x, return_attn=False):
        B = x.shape[0]

        x = self.patch_embd(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_embd(x)

        attn_maps = []

        for block in self.blocks:
            if return_attn:
                x, attn = block(x, return_attn=True)
                attn_maps.append(attn)
            else:
                x = block(x)

        x = self.ln_f(x)

        logits = self.head(x[:, 0])

        if return_attn:
            return logits, attn_maps
        return logits

    def training_step(self, batch, batch_idx):
        img, target = batch
        logits = self(img)
        loss = F.cross_entropy(logits, target)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, target = batch
        logits = self(img)
        loss = F.cross_entropy(logits, target)
        acc = (logits.argmax(dim=-1) == target).float().mean()

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and not name.endswith("bias"):
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.lr,
            betas=self.config.training.betas,
        )

        scheduler = get_scheduler(
            name=self.config.training.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.config.training.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
