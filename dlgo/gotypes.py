from enum import Enum
from collections import namedtuple


class Player(Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


class Point:
    """
    for immutability, and hashability
    also, access it as point.row and point.col instead of point[0] and point[1]
    """

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def neighbbors(self):
        return [
            Point(self.row - 1, self.col),  # up
            Point(self.row + 1, self.col),  # down
            Point(self.row, self.col - 1),  # left
            Point(self.row, self.col + 1),  # right
        ]


if __name__ == "__main__":
    assert Player.black.other == Player.white
    assert Player.white.other == Player.black
    print("test passed")
