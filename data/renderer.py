from PIL import Image, ImageDraw


class MazeRenderer:
    def __init__(self, grid_size=10, image_size=(100, 100)):
        self.image_size = image_size
        self.grid_size = grid_size

        self.cell_size = image_size[0] // grid_size

    def render(self, maze, agent_pos, exit_pos):
        img = Image.new("RGB", self.image_size, (255, 255, 255))
        draw = ImageDraw.Draw(img)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if maze[r, c] == 1:
                    self._draw_cell(draw, r, c, (0, 0, 0))

        self._draw_cell(draw, exit_pos[0], exit_pos[1], (0, 255, 0))
        self._draw_cell(draw, agent_pos[0], agent_pos[1], (255, 0, 0))
        return img

    def _draw_cell(self, draw, r, c, color):
        top = r * self.cell_size
        left = c * self.cell_size
        bottom = top + self.cell_size
        right = left + self.cell_size

        draw.rectangle([left, top, right, bottom], fill=color)
