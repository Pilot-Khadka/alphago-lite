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
        """
        Merge current stone with an existing chain
        only when they are of same color
        """
        # only stones of same color can form a chain
        assert go_string.color == self.color, "Can merge only with same color"

        # set union stonew
        all_stones = self.stones | go_string.stones
        all_liberties = (self.liberties | go_string.liberties) - all_stones
        return GoString(self.color, all_stones, all_liberties)

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
        assert self.is_on_board(point), "Move is outside the board"
        assert self.is_empty(point), "Point already occupied"

        same_color_strigs = set()
        opposite_color_strings = set()
        liberties = set()

        #  look at neighbors of the point
        for neighbor in point.neighbors():
            if not self.is_on_board(neighbor):
                continue

            if self.is_empty(neighbor):
                liberties.add(neighbor)
            else:
                neighbor_group = self._grid.get(neighbor)
                if neighbor_group.color == player:
                    same_color_strigs.add(neighbor_group)
                else:
                    opposite_color_strings.add(neighbor_group)

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

    def is_on_board(self, point):
        """Check if point is withing board range"""
        return 1 <= point.row <= self.size and 1 <= point.col <= self.size

    def is_empty(self, point):
        """check if point has no stone"""
        return point not in self._grid

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


class GameState:
    def __init__(self, board, next_player, previous_state, last_move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous_state
        self.last_move = last_move

    def apply_move(self, move):
        """
        Return new gamestate after applying given move
        """
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board

        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        """
        Create new game with given boardsize
        """
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return GameState(board, Player.black, None, None)

    def is_over(self):
        """
        Return true if game has ended by resignation or two passes
        """
        if self.last_move is None:
            return False

        if self.last_move.is_resign:
            return True

        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    def is_move_self_capture(self, player, move):
        if not move.is_play:
            return False

        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        return new_string.num_liberties == 0


if __name__ == "__main__":
    s1 = GoString("black", {(1, 1), (1, 2)}, {(0, 1), (1, 3)})
    s2 = GoString("black", {(1, 1), (1, 2)}, {(0, 1), (1, 3)})

    print(s1 == s2)  # True — because __eq__ is defined
