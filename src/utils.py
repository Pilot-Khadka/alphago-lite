import tkinter as tk
from dlgo import gotypes
from dlgo.gotypes import Player

cols = "ABCDEFGHIJKLMNOPQRST"
stone_to_char = {None: ".", Player.black: "X", Player.white: "O"}
EMPTY = None
BLACK = Player.black
WHITE = Player.white


def print_move(player, move):
    if move.is_pass:
        move_str = "passes"
    elif move.is_resign:
        move_str = "resigns"
    else:
        move_str = "%s%d" % (cols[move.point.col - 1], move.point.row)
        print("%s %s" % (player, move_str))


class GoBoardDisplay:
    def __init__(self, board_size=19):
        self.board_size = board_size

        # gui setup
        self.root = tk.Tk()
        self.root.title("Go Board Display")
        self.root.resizable(False, False)

        # canvas for the board
        self.cell_size = 25
        self.margin = 30
        canvas_size = (board_size - 1) * self.cell_size + 2 * self.margin

        self.canvas = tk.Canvas(
            self.root,
            width=canvas_size,
            height=canvas_size,
            bg="#dcb35c",
            highlightthickness=0,
        )
        self.canvas.pack(pady=10)

        # store star points for 19x19 board
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

        # init empty board
        self.board_grid = [
            [EMPTY for _ in range(board_size)] for _ in range(board_size)
        ]
        self.draw_board()

    def board_to_pixel(self, row, col):
        x = col * self.cell_size + self.margin
        y = row * self.cell_size + self.margin
        return x, y

    def pixel_to_board(self, x, y):
        """convert pixel coordinates to board coordinates"""
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        return row, col

    def on_click(self, event):
        """Handle mouse click on the board"""
        if self.click_callback:
            row, col = self.pixel_to_board(event.x, event.y)
            # Convert to 1-based indexing for game logic
            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                self.click_callback(row + 1, col + 1)

    def draw_board(self):
        self.canvas.delete("all")

        # draw grid lines
        for i in range(self.board_size):
            # vertical lines
            x = i * self.cell_size + self.margin
            self.canvas.create_line(
                x,
                self.margin,
                x,
                (self.board_size - 1) * self.cell_size + self.margin,
                fill="black",
                width=1,
            )

            # horizontal lines
            y = i * self.cell_size + self.margin
            self.canvas.create_line(
                self.margin,
                y,
                (self.board_size - 1) * self.cell_size + self.margin,
                y,
                fill="black",
                width=1,
            )

        # draw star points
        for row, col in self.star_points:
            x, y = self.board_to_pixel(row, col)
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black")

        # draw stones
        stone_radius = self.cell_size // 2 - 2
        for row in range(self.board_size):
            for col in range(self.board_size):
                stone = self.board_grid[row][col]
                if stone is not None:  # Not empty
                    x, y = self.board_to_pixel(row, col)

                    if stone == BLACK:
                        self.canvas.create_oval(
                            x - stone_radius,
                            y - stone_radius,
                            x + stone_radius,
                            y + stone_radius,
                            fill="black",
                            outline="black",
                        )
                    elif stone == WHITE:
                        self.canvas.create_oval(
                            x - stone_radius,
                            y - stone_radius,
                            x + stone_radius,
                            y + stone_radius,
                            fill="white",
                            outline="black",
                            width=2,
                        )

    def update_board(self, board):
        for row in range(1, board.size + 1):
            for col in range(1, board.size + 1):
                stone = board.get(gotypes.Point(row=row, col=col))
                # Convert to 0-based indexing for internal storage
                self.board_grid[row - 1][col - 1] = stone

        self.draw_board()
        self.root.update()

    def show(self):
        self.root.update()

    def run(self):
        self.root.mainloop()

    def close(self):
        self.root.destroy()
