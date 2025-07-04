import copy
from dlgo.gotypes import Player


class Move:
    """
    A move can be:
    - playing at a point,
    - passing the turn,
    - resigning.
    only one must be chosen
    """

    def __init__(self, point=None, is_pass=False, is_resign=False):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = self.point is not None
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def pass_turn(cls, point):
        return Move(is_pass=True)

    @classmethod
    def resign(cls, point):
        return Move(is_resign=True)


class GoString:
    """
    Tracks entire chain of connected stones
    """

    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = set(stones)
        self.liberties = set(liberties)

    def remove_liberty(self, point):
        self.liberties.remove(point)

    def add_liberty(self, point):
        self.liberties.add(point)

    def merged_with(self, go_string):
        # only stones of same color can form a chain
        assert go_string.color == self.color

        # set union stonew
        combined_stones = self.stones | go_string.stones

        return GoString(
            self.color,
            combined_stones,
            # combine liberties, but remove the currently occupied place
            (self.liberties | go_string.liberties) - combined_stones,
        )

    @property
    def num_liberties(self):
        return len(self.liberties)

    def __eq__(self, other):
        if not isinstance(other, GoString):
            return False
        return (
            self.color == other.color
            and self.stones == other.stones
            and self.liberties == other.liberties
        )


class Board:
    def __init__(self, size: int = 19):
        self.size = size
        self._grid = {}

    def place_stone(self, player, point):
        assert self.is_on_grid(point), "Move is outside the board"
        assert self._grid.get(point) is None, "Point is already occuped"

        same_color_strigs = set()
        opposite_color_strings = set()
        liberties = set()

        #  look at neighbors of the point
        for neighbor in point.neighbors():
            if not self.is_on_grid(neighbor):
                continue

            neighbor_string = self._grid.get(neighbor)

            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                same_color_strigs.add(neighbor_string)
            else:
                opposite_color_strings.add(neighbor_string)

        # create new group with the placed stone
        new_string = GoString(player, [point], liberties)

        # merge the group with neighboring same color group
        for string in same_color_strigs:
            new_string = new_string.merged_with(string)

        # update the board to point all stones in group to merget stone
        for stone in new_string.stones:
            self._grid[stone] = new_string

        # reduce liberties of opposite color group,
        # remove if captured
        for string in opposite_color_strings:
            string.remove_liberty(point)
            if string.num_liberties == 0:
                self._remove_string(string)

    def is_on_grid(self, point):
        """Check if point is withing board range"""
        return 1 <= point.row <= self.size and 1 <= point.col <= self.size

    def get(self, point):
        """
        Return color of string at the point, None if empty
        """
        string = self._grid.get(point)
        return string.color if string else None

    def get_go_string(self, point):
        """
        Return full gostring at the point, none if empty
        """
        self._grid.get(point)

    def _remove_string(self, string):
        """
        Remove captured string from board and update neighbors liberties
        """
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string and neighbor_string is not string:
                    neighbor_string.add_liberty(point)
            self._grid[point] = None


if __name__ == "__main__":
    s1 = GoString("black", {(1, 1), (1, 2)}, {(0, 1), (1, 3)})
    s2 = GoString("black", {(1, 1), (1, 2)}, {(0, 1), (1, 3)})

    print(s1 == s2)  # True — because __eq__ is defined
