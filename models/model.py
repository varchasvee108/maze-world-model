import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from models.layers import PatchEmbedding, PositionEmbedding
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

        grid_size = config.input_data.image_size[0] // config.model.patch_size

        self.pos_embd = PositionEmbedding(
            num_patches=grid_size * grid_size,
            n_embd=config.model.n_embd,
            grid_size=grid_size,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model.n_embd,
            nhead=config.model.num_heads,
            dim_feedforward=config.model.n_embd * config.model.dim_ratio,
            dropout=config.model.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.model.num_layers,
        )

        self.ln_f = nn.LayerNorm(config.model.n_embd)
        self.head = nn.Linear(config.model.n_embd, 4)

    def forward(self, x):

        x = self.patch_embd(x)
        x = self.pos_embd(x)
        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.head(self.ln_f(x))

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
