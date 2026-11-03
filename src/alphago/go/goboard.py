from . import zobrist
from .gotypes import Player, Point
from .scoring import DEFAULT_KOMI, compute_game_result

NEIGHBOR_TABLES = {}


def neighbor_table(size):
    """
    on-board neighbors of every point, built once per board size
    iteration order is row-major, so it doubles as the list of all points
    """
    if size not in NEIGHBOR_TABLES:
        table = {}
        for row in range(1, size + 1):
            for col in range(1, size + 1):
                point = Point(row, col)
                table[point] = tuple(
                    nb
                    for nb in point.neighbors()
                    if 1 <= nb.row <= size and 1 <= nb.col <= size
                )
        NEIGHBOR_TABLES[size] = table
    return NEIGHBOR_TABLES[size]


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
        self._neighbors = neighbor_table(size)

    def place_stone(self, player, point):
        assert self.is_on_board(point), "Move is outside the board"
        assert self.is_empty(point), "Point already occupied"

        # keyed by id so a group touching the point twice is handled once
        same_color_strings = {}
        opposite_color_strings = {}
        liberties = []

        #  look at neighbors of the point
        for neighbor in self._neighbors[point]:
            neighbor_group = self._grid.get(neighbor)
            if neighbor_group is None:
                liberties.append(neighbor)
            elif neighbor_group.color == player:
                same_color_strings[id(neighbor_group)] = neighbor_group
            else:
                opposite_color_strings[id(neighbor_group)] = neighbor_group

        # create new group with the placed stone
        new_string = GoString(player, [point], liberties)

        # merge the group with neighboring same color group
        for string in same_color_strings.values():
            new_string = new_string.merged_with(string)

        # update the board to point all stones in group to merget stone
        for stone in new_string.stones:
            self._grid[stone] = new_string

        self._hash ^= zobrist.HASH_CODE[point, player]

        # reduce liberties of opposite color group,
        # remove if captured
        for string in opposite_color_strings.values():
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
            for neighbor in self._neighbors[point]:
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
    def __init__(self, board, next_player, previous_state, last_move, komi=None):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous_state
        self.last_move = last_move
        if komi is None:
            komi = previous_state.komi if previous_state else DEFAULT_KOMI
        self.komi = komi

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
        """
        GoStrings are immutable and place_stone only swaps grid entries,
        so a shallow copy of the grid is enough
        """
        new_board = Board(self.board.size)
        new_board._grid = self.board._grid.copy()
        new_board._hash = self.board._hash
        return new_board

    @classmethod
    def new_game(cls, board_size, komi=DEFAULT_KOMI):
        """
        Create new game with given boardsize
        """
        board = Board(board_size)
        return GameState(board, Player.black, None, None, komi=komi)

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

        return self.result().winner

    def result(self):
        """
        Area score of the current position, also usable before the game ends
        """
        return compute_game_result(self.board, self.komi)

    def _evaluate_play(self, player, point):
        """
        read the outcome of playing at point from the neighboring groups
        instead of placing the stone on a board copy
        returns (is_suicide, zobrist hash of the resulting board)
        """
        board = self.board
        has_liberty = False
        captured = {}
        for neighbor in board._neighbors[point]:
            string = board._grid.get(neighbor)
            if string is None:
                has_liberty = True
            elif string.color == player:
                if string.num_liberties > 1:
                    has_liberty = True
            elif string.num_liberties == 1:
                captured[id(string)] = string

        new_hash = board._hash ^ zobrist.HASH_CODE[point, player]
        for string in captured.values():
            for stone in string.stones:
                new_hash ^= zobrist.HASH_CODE[stone, string.color]

        return not has_liberty and not captured, new_hash

    def _is_legal_play(self, point):
        is_suicide, new_hash = self._evaluate_play(self.next_player, point)
        if is_suicide:
            return False
        return (self.next_player.other, new_hash) not in self.previous_positions

    def is_move_self_capture(self, player, move):
        """
        if playing the move leaves its group with zero liberties,
        i.e no free neighborhood, it is not a valid move
        """
        if not move.is_play:
            return False
        return self._evaluate_play(player, move.point)[0]

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

        new_hash = self._evaluate_play(player, move.point)[1]
        return (player.other, new_hash) in self.previous_positions

    def is_valid_move(self, move):
        if self.is_over():
            return False

        if move.is_pass or move.is_resign:
            return True

        if not self.board.is_on_board(move.point):
            return False
        if not self.board.is_empty(move.point):
            return False
        return self._is_legal_play(move.point)

    def legal_moves(self):
        moves = []
        if not self.is_over():
            grid = self.board._grid
            for point in self.board._neighbors:
                if point not in grid and self._is_legal_play(point):
                    moves.append(Move.play(point))

        moves.append(Move.pass_turn())
        moves.append(Move.resign())

        return moves
