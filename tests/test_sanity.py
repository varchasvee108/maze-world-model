import torch
import torch.nn.functional as F
from core.config import Config
from models.model import MazeTransformer
from pathlib import Path
import torch.nn as nn


def test_sanity():
    config = Config.load(Path("config/base.toml"))
    model = MazeTransformer(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    img = torch.randn(1, 3, 132, 132)
    target = torch.tensor([2])
    model.train()
    initial_loss = None

    for step in range(200):
        optimizer.zero_grad()
        logits = model(img)
        loss = loss_fn(logits, target)

        if step == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}, Loss: {loss.item():.4f}")

    assert loss.item() < initial_loss * 0.05, "Loss did not decrease"
    print("Sanity test passed")


if __name__ == "__main__":
    try:
        test_sanity()
    except AssertionError as e:
        print(f"Test failed: {str(e)}")
    except Exception as e:
        print(f"Test failed: {str(e)}")
