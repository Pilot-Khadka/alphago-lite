import numpy as np


from ..encoders.base import Encoder
from ..goboard import Move, Point


class SevenPlaneEncoder(Encoder):
    def __init__(self, board_size):
        self.board_size = board_size
        self.num_planes = 7

    # pyrefly: ignore [bad-override]
    def name(self):
        return "sevenplane"

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        base_plane = {game_state.next_player: 0, game_state.next_player.other: 3}
        for row in range(self.board_size):
            for col in range(self.board_size):
                p = Point(row=row + 1, col=col + 1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    # last plane encodes point that cant be played becaus eof ko
                    if game_state.does_move_violate_ko(
                        game_state.next_player, Move.play(p)
                    ):
                        board_tensor[6][row][col] = 1

                    else:
                        # pyrefly: ignore [missing-attribute]
                        liberty_plane = min(3, go_string.num_liberties) - 1
                        # pyrefly: ignore [missing-attribute]
                        liberty_plane += base_plane[go_string.color]
        return board_tensor

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


def create(board_size):
    return SevenPlaneEncoder(board_size)
