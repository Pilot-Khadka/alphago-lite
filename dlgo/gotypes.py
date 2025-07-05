from enum import Enum


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

    def neighbors(self):
        return [
            Point(self.row - 1, self.col),  # up
            Point(self.row + 1, self.col),  # down
            Point(self.row, self.col - 1),  # left
            Point(self.row, self.col + 1),  # right
        ]

    def __eq__(self, other):
        """Two points are equal if they have the same row and col"""
        return (
            isinstance(
                other, Point) and self.row == other.row and self.col == other.col
        )

    def __hash__(self):
        """Make Point hashable so it can be used as a dictionary key"""
        return hash((self.row, self.col))

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
