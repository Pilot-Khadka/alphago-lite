from enum import Enum
from typing import NamedTuple


class Player(Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


class Point(NamedTuple):
    """
    for immutability, and hashability
    also, access it as point.row and point.col instead of point[0] and point[1]
    """

    row: int
    col: int

    def neighbors(self):
        return [
            Point(self.row - 1, self.col),  # up
            Point(self.row + 1, self.col),  # down
            Point(self.row, self.col - 1),  # left
            Point(self.row, self.col + 1),  # right
        ]

    def __str__(self):
        """String representation for debugging"""
        return f"Point({self.row},{self.col})"

    def __repr__(self):
        """String representation for debugging"""
        return f"Point({self.row},{self.col})"


if __name__ == "__main__":
    assert Player.black.other == Player.white
    assert Player.white.other == Player.black
    print("test passed")
