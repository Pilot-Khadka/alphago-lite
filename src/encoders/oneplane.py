import numpy as np

from ..encoders.base import Encoder
from ..goboard import Point


def create(board_size):
    return OnePlaneEncoder(board_size)


class OnePlaneEncoder(Encoder):
    def __init__(self, board_size):
        if isinstance(board_size, tuple):
            assert board_size[0] == board_size[1], "Only square boards supported"
            board_size = board_size[0]

        self.board_size = board_size
        self.num_planes = 1

    # pyrefly: ignore [bad-override]
    def name(self):
        return "oneplane"

    def encode(self, game_state):
        board_matrix = np.zeros(self.shape(), dtype=np.float32)
        next_player = game_state.next_player

        for row in range(self.board_size):
            for col in range(self.board_size):
                point = Point(row=row + 1, col=col + 1)
                go_string = game_state.board.get_go_string(point)

                if go_string is None:
                    continue
                base_value = 1.0 if go_string.color == next_player else -1.0
                board_matrix[0, row, col] = base_value

        return board_matrix

    def encode_point(self, point):
        return self.board_size * (point.row - 1) + (point.col - 1)

    # pyrefly: ignore [bad-override]
    def decode_point_index(self, index):
        row = index // self.board_size
        col = index % self.board_size
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_size * self.board_size

    # pyrefly: ignore [bad-override]
    def shape(self):
        return (self.num_planes, self.board_size, self.board_size)
