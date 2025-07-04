import Enum
import tkinter as tk
from dataclasses import dataclass
from typing import List


class Player(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def opponent(self):
        if self == Player.BLACK:
            return Player.WHITE
        elif self == Player.WHITE:
            return Player.BLACK
        return Player.EMPTY


@dataclass(frozen=True)
class Point:
    """Immutable point representation - crucial for hashing and AlphaGo features"""

    row: int
    col: int

    def neighbors(self, board_size: int) -> List["Point"]:
        """Get valid neighboring points"""
        neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = self.row + dr, self.col + dc
            if 0 <= nr < board_size and 0 <= nc < board_size:
                neighbors.append(Point(nr, nc))
        return neighbors

    def __hash__(self):
        return hash((self.row, self.col))


class GO:
    def __init__(self, board_size=19):
        self.board_size = board_size

        # ui setup
        self.root = tk.Tk()
        self.root.title("Go game")
        self.root.resizable(False, False)

        self.cell_size = 25
        self.margin = 25
        canvas_size = (board_size - 1) * self.cell_size + 2 * self.margin

        self.canvas = tk.Canvas(
            self.root,
            width=canvas_size,
            height=canvas_size,
            bg="#dcb35c",
            highlightthickness=0,
        )
        self.canvas.pack(pady=10)
        self.star_points = []
        if board_size == 19:
            self.star_points = [
                (3, 3),
                (3, 9),
                (3, 15),
                (9, 3),
                (9, 9),
                (9, 15),
                (15, 3),
                (15, 9),
                (15, 15),
            ]

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")

        for i in range(self.board_size):
            # vertical line
            x = i * self.cell_size + self.margin
            self.canvas.create_line(
                x,
                self.margin,
                x,
                (self.board_size - 1) * self.cell_size + self.margin,
                fill="black",
                width=1,
            )

            # horizontal linw
            y = i * self.cell_size + self.margin
            self.canvas.create_line(
                self.margin,
                y,
                (self.board_size - 1) * self.cell_size + self.margin,
                y,
                fill="black",
                width=1,
            )

        for row, col in self.star_points:
            x, y = self.board_to_pixel(row, col)
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black")

    def board_to_pixel(self, row, col):
        x = col * self.cell_size + self.margin
        y = row * self.cell_size + self.margin
        return x, y

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    game = GO()
    game.run()
