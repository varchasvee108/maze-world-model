from data.generator import MazeTransitionGenerator
from data.renderer import MazeRenderer

gen = MazeTransitionGenerator(grid_size=11)
rend = MazeRenderer(grid_size=11, image_size=(440, 440))

samples = gen.generate_policy_samples()

if samples:
    print(f"These are the samples: {len(samples)}")
    first_step = samples[0]

    image = rend.render(
        maze=first_step["maze"],
        agent_pos=first_step["start"],
        exit_pos=first_step["exit"],
    )

    image.save("test_maze.png")

    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    print(f"Start: {first_step['start']} | Exit: {first_step['exit']}")
    print(f"Action: {action_names[first_step['label']]}")
