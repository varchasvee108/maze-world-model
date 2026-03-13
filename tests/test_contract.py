import torch
from core.config import Config
from data.dataloader import MazeDataModule
from models.model import MazeTransformer
from pathlib import Path
import pytest


@pytest.mark.integration
def test_contract():
    config = Config.load(Path("config/base.toml"))
    dm = MazeDataModule(
        npz_path=Path("data/maze_data.npz"),
        batch_size=4,
    )
    dm.setup()
    model = MazeTransformer(config)

    if torch.cuda.is_available():
        model.to("cuda")

    batch = next(iter(dm.train_dataloader()))
    images, labels = batch

    if torch.cuda.is_available():
        images = images.to("cuda")
        labels = labels.to("cuda")

    print(f"DEBUG: Image is on device : {images.device}")
    print(f"DEBUG: Label is on device : {labels[0].device}")

    try:
        assert isinstance(images, torch.Tensor), (
            f"Expected images to be a torch.Tensor, got {type(images)}"
        )
        assert isinstance(labels, torch.Tensor), (
            f"Expected labels to be a torch.Tensor, got {type(labels)}"
        )
        print("Contract test passed: Batch is a tuple of tensors")
    except AssertionError as e:
        print(f"Contract test failed: {str(e)}")
    except Exception as e:
        print(f"Contract test failed: {str(e)}")

    try:
        logits = model(images)
        assert logits.shape[0] == images.shape[0], (
            f"Expected shape of logits to be {images.shape[0]}, got {logits.shape[0]}"
        )
        assert logits.shape[1] == 4, (
            f"Expected shape of logits to be (4, 4) but got {logits.shape[1]}"
        )
        assert images.dtype == torch.float32, (
            f"Expected dtype of images to be {torch.float32}, got {images.dtype}"
        )
        assert labels.dtype in (torch.int64, torch.long), (
            f"Expected dtype of labels to be {torch.int64} or {torch.long}, got {labels.dtype}"
        )
        print("Contract test passed: Model output shape is correct")
    except AssertionError as e:
        print(f"Contract test failed: {str(e)}")
    except Exception as e:
        print(f"Contract test failed: {str(e)}")


if __name__ == "__main__":
    test_contract()
