import numpy as np

from ..encoders.base import Encoder
from ..go.goboard import Point


def create(board_size):
    return OnePlaneEncoder(board_size)


class OnePlaneEncoder(Encoder):
    def __init__(self, board_size):
        if isinstance(board_size, tuple):
            assert board_size[0] == board_size[1], "Only square boards supported"
            board_size = board_size[0]

        self.board_size = board_size
        self.num_planes = 1

    def name(self):
        return "oneplane"

    def encode(self, game_state):
        board_matrix = np.zeros(self.shape(), dtype=np.float32)
        next_player = game_state.next_player

        for point, go_string in game_state.board._grid.items():
            row, col = point.row - 1, point.col - 1
            board_matrix[0, row, col] = 1.0 if go_string.color == next_player else -1.0

        return board_matrix

    def encode_batch(self, game_states):
        batch = np.zeros((len(game_states), *self.shape()), dtype=np.float32)
        for i, gs in enumerate(game_states):
            next_player = gs.next_player
            for point, go_string in gs.board._grid.items():
                row, col = point.row - 1, point.col - 1
                batch[i, 0, row, col] = 1.0 if go_string.color == next_player else -1.0
        return batch

    def encode_point(self, point):
        return self.board_size * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index):
        row = index // self.board_size
        col = index % self.board_size
        return Point(row=row + 1, col=col + 1)

    def num_points(self):
        return self.board_size * self.board_size

    def shape(self):
        return (self.num_planes, self.board_size, self.board_size)
