from data.generator import MazeTransitionGenerator
from data.renderer import MazeRenderer

gen = MazeTransitionGenerator(grid_size=10)
rend = MazeRenderer(image_size=(300, 300), grid_size=10)

# 2. Get data
transitions = gen.get_maze_transitions()
first_step = transitions[0]

# 3. Render
image = rend.render(
    maze=first_step["maze"],
    agent_pos=first_step["start"],
    exit_pos=first_step["exit"],
)

# 4. Save and look at it!
image.save("test_maze.png")
print(f"Action tried: {first_step['action']} | Outcome label: {first_step['label']}")
