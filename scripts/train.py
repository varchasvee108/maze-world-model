import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from core.config import Config
from data.dataloader import MazeDataModule
from models.model import MazeTransformer


def main():
    parser = argparse.ArgumentParser(description="Train a maze transformer")
    parser.add_argument(
        "--config", type=str, default="config/base.toml", help="Path to the config file"
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to the checkpoint file"
    )
    args = parser.parse_args()

    config = Config.load(args.config)

    L.seed_everything(config.input_data.seed)

    datamodule = MazeDataModule(
        npz_path="data/maze_data.npz", batch_size=config.input_data.batch_size
    )

    model = MazeTransformer(config=config)

    logger = WandbLogger(
        project=config.logging.project_name,
        name=config.project.experiment_name,
        save_dir="outputs/wandb",
    )

    checkpoint = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename="maze-{step}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = L.Trainer(
        max_steps=config.training.max_steps,
        val_check_interval=config.training.eval_interval,
        precision="16-mixed",
        gradient_clip_val=config.training.gradient_clipping,
        accelerator="auto",
        logger=logger,
        callbacks=[
            checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
        log_every_n_steps=config.training.log_interval,
        check_val_every_n_epoch=1000,
        limit_val_batches=0.4,
    )

    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=args.ckpt_path, weights_only=False
    )


if __name__ == "__main__":
    main()
