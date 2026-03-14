import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import cv2
import random
import json

from core.config import Config
from models.model import MazeTransformer
from maze_dataset.dataset import MazeVisionDataset
from data.renderer import MazeRenderer


VISUAL_DIR = Path("visuals")
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


def compute_attention_rollout(attn_maps):

    result = None

    for attn in attn_maps:
        attn = attn[0]
        attn = attn.detach().cpu().numpy()

        T = attn.shape[0]

        identity = np.eye(T)

        attn = attn + identity
        attn = attn / attn.sum(axis=-1, keepdims=True)

        if result is None:
            result = attn
        else:
            result = attn @ result

    return result


def extract_rollout_grid(rollout, grid_size=11):

    cls_attention = rollout[0, 1:]

    grid = cls_attention.reshape(grid_size, grid_size)

    if grid.max() > 0:
        grid = grid / grid.max()

    return grid


def upscale_attention(grid, image_size=132):

    return cv2.resize(
        grid,
        (image_size, image_size),
        interpolation=cv2.INTER_CUBIC,
    )


def overlay_attention(image, attention_map, save_path):

    img = np.array(image)

    fig = go.Figure()

    fig.add_trace(go.Image(z=img))

    fig.add_trace(
        go.Heatmap(
            z=attention_map,
            colorscale="Inferno",
            opacity=0.55,
            showscale=False,
        )
    )

    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.write_image(save_path)


def detect_three_oscillation(recent_positions):

    if len(recent_positions) < 6:
        return False

    a, b, c, d, e, f = recent_positions[-6:]

    return a == c == e and b == d == f


def run_single_rollout(model, dataset, rollout_id, device):

    maze_idx = random.randint(0, len(dataset.mazes) - 1)

    maze = dataset.mazes[maze_idx]
    exit_pos = tuple(int(x) for x in dataset.exits[maze_idx])

    maze_samples = dataset.samples[dataset.samples[:, 0] == maze_idx]

    sample = maze_samples[random.randint(0, len(maze_samples) - 1)]

    agent = (int(sample[1]), int(sample[2]))
    start_pos = agent

    renderer = MazeRenderer(grid_size=11)

    rollout_dir = VISUAL_DIR / f"maze_{rollout_id}"
    rollout_dir.mkdir(exist_ok=True)

    frame_dir = rollout_dir / "frames"
    frame_dir.mkdir(exist_ok=True)

    recent_positions = []
    path = []

    print(f"\nRollout {rollout_id}")
    print(f"Maze index: {maze_idx}, start: {agent}, exit: {exit_pos}")

    for step in range(40):
        path.append(agent)
        recent_positions.append(agent)

        if detect_three_oscillation(recent_positions):
            print("3 oscillations detected — stopping rollout")
            break

        image = renderer.render(maze, agent, exit_pos)

        tensor = dataset.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, attn_maps = model(tensor, return_attn=True)

        action = torch.argmax(logits, dim=-1).item()

        rollout = compute_attention_rollout(attn_maps)

        grid = extract_rollout_grid(rollout)

        grid = upscale_attention(grid)

        save_path = frame_dir / f"step_{step}.png"

        overlay_attention(image, grid, save_path)

        dx, dy = ACTIONS[action]

        new_x = agent[0] + dx
        new_y = agent[1] + dy

        if (
            0 <= new_x < maze.shape[0]
            and 0 <= new_y < maze.shape[1]
            and maze[new_x, new_y] == 0
        ):
            agent = (new_x, new_y)

        if agent == exit_pos:
            path.append(agent)
            print(f"Reached exit in {step + 1} steps")
            break

    rollout_data = {
        "grid_size": int(maze.shape[0]),
        # convert entire maze to Python ints
        "maze": [[int(cell) for cell in row] for row in maze],
        "start": [int(start_pos[0]), int(start_pos[1])],
        "exit": [int(exit_pos[0]), int(exit_pos[1])],
        "path": [[int(p[0]), int(p[1])] for p in path],
    }

    with open(rollout_dir / "rollout.json", "w") as f:
        json.dump(rollout_data, f, indent=2)


def rollout_visualization():

    config = Config.load("config/base.toml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MazeTransformer.load_from_checkpoint(
        "outputs/checkpoints/best_final_model.ckpt",
        config=config,
        weights_only=False,
    ).to(device)

    model.eval()

    dataset = MazeVisionDataset("data/maze_data.npz")

    NUM_ROLLOUTS = 10

    for i in range(NUM_ROLLOUTS):
        run_single_rollout(
            model=model,
            dataset=dataset,
            rollout_id=i + 1,
            device=device,
        )


if __name__ == "__main__":
    rollout_visualization()
