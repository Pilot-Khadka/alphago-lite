# pyrefly: ignore [missing-import]
import src.zobrist as zobrist
from .gotypes import Player, Point


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
    def pass_turn(cls):
        return Move(is_pass=True)

    @classmethod
    def resign(cls):
        return Move(is_resign=True)


class GoString:
    """
    Tracks entire chain of connected stones
    """

    def __init__(self, color, stones, liberties):
        self.color = color
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)

    def without_liberty(self, point):
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)

    def with_liberty(self, point):
        new_liberties = self.liberties | set([point])
        return GoString(self.color, self.stones, new_liberties)

    def merged_with(self, go_string):
        """
        Merge current stone with an existing chain
        if  they are of same color
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
        self._hash = zobrist.EMPTY_BOARD

    def place_stone(self, player, point):
        assert self.is_on_board(point), "Move is outside the board"
        assert self.is_empty(point), "Point already occupied"

        same_color_strigs = []
        opposite_color_strings = []
        liberties = []

        #  look at neighbors of the point
        for neighbor in point.neighbors():
            if not self.is_on_board(neighbor):
                continue

            if self.is_empty(neighbor):
                liberties.append(neighbor)
            else:
                neighbor_group = self._grid.get(neighbor)
                # pyrefly: ignore [missing-attribute]
                if neighbor_group.color == player:
                    same_color_strigs.append(neighbor_group)
                else:
                    opposite_color_strings.append(neighbor_group)

        # create new group with the placed stone
        new_string = GoString(player, [point], liberties)

        # merge the group with neighboring same color group
        for string in same_color_strigs:
            new_string = new_string.merged_with(string)

        # update the board to point all stones in group to merget stone
        for stone in new_string.stones:
            self._grid[stone] = new_string

        self._hash ^= zobrist.HASH_CODE[point, player]

        # reduce liberties of opposite color group,
        # remove if captured
        for string in opposite_color_strings:
            # pyrefly: ignore [missing-attribute]
            replacement = string.without_liberty(point)
            if replacement.num_liberties:
                self._replace_string(replacement)
            else:
                self._remove_string(string)

    def _replace_string(self, new_string):
        for point in new_string.stones:
            self._grid[point] = new_string

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
        return self._grid.get(point)

    def _remove_string(self, string):
        """
        Remove captured string from board and update neighbors liberties
        """
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string and neighbor_string is not string:
                    updated_string = neighbor_string.with_liberty(point)
                    self._replace_string(updated_string)

            if point in self._grid:
                del self._grid[point]
                self._hash ^= zobrist.HASH_CODE[point, string.color]

    def zobrist_hash(self):
        return self._hash


class GameState:
    def __init__(self, board, next_player, previous_state, last_move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous_state
        self.last_move = last_move

        if previous_state is None:
            self.previous_positions = set()
        else:
            self.previous_positions = previous_state.previous_positions.copy()
            self.previous_positions.add(
                (previous_state.next_player, previous_state.board.zobrist_hash())
            )

    def apply_move(self, move):
        """
        Return new gamestate after applying given move
        """
        if move.is_play:
            next_board = self._copy_board()
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board

        return GameState(next_board, self.next_player.other, self, move)

    def _copy_board(self):
        """more efficient board copying"""
        new_board = Board(self.board.size)
        new_board._grid = {}
        new_board._hash = self.board._hash

        string_map = {}
        for point, string in self.board._grid.items():
            string_id = id(string)
            if string_id not in string_map:
                string_map[string_id] = GoString(
                    string.color,
                    frozenset(string.stones),
                    frozenset(string.liberties),
                )
            new_board._grid[point] = string_map[string_id]

        return new_board

    @classmethod
    def new_game(cls, board_size):
        """
        Create new game with given boardsize
        """
        board = Board(board_size)
        return GameState(board, Player.black, None, None)

    def is_over(self):
        """
        Return true if game has ended by resignation or two passes
        """
        if self.last_move is None:
            return False

        if self.last_move.is_resign:
            return True
        if self.previous_state is None:
            return False

        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    def winner(self):
        """
        Determine winner of the game
        Returns Player.black, Player.white, or None for draw
        """
        if not self.is_over():
            return None

        # If someone resigned, the other player wins
        if self.last_move and self.last_move.is_resign:
            return self.next_player  # The player who didn't resign

        return self._count_stones_winner()

    def _count_stones_winner(self):
        black_stones = 0
        white_stones = 0

        for r in range(1, self.board.size + 1):
            for c in range(1, self.board.size + 1):
                point = Point(row=r, col=c)
                stone = self.board.get(point)
                if stone == Player.black:
                    black_stones += 1
                elif stone == Player.white:
                    white_stones += 1

        if black_stones > white_stones:
            return Player.black
        elif white_stones > black_stones:
            return Player.white
        else:
            return None  # Draw

    def is_move_self_capture(self, player, move):
        """
        simulate playing current move
        if it results in zero liberties,
        i.e no free neighborhood, it is not a valid move
        """
        if not move.is_play:
            return False

        next_board = self._copy_board()
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        # pyrefly: ignore [missing-attribute]
        return new_string.num_liberties == 0

    @property
    def situation(self):
        return (self.next_player, self.board)

    def does_move_violate_ko(self, player, move):
        """
        ko is a condition when a sucessive moves
        results in the board being in previous position
        eg:
        black captures a piece of white, white can do the same
        board positionn is the exact same as two moves previously
        """
        if not move.is_play:
            return False

        temp_board = self._copy_board()
        temp_board.place_stone(player, move.point)
        next_situation = (player.other, temp_board.zobrist_hash())
        return next_situation in self.previous_positions

    def is_valid_move(self, move):
        if self.is_over():
            return False

        if move.is_pass or move.is_resign:
            return True

        # check if the board is empty
        if self.board.get(move.point) is not None:
            return False

        is_empty = self.board.get(move.point) is None
        not_suicide = not self.is_move_self_capture(self.next_player, move)
        not_ko = not self.does_move_violate_ko(self.next_player, move)

        return is_empty and not_suicide and not_ko
