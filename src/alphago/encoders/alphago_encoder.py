import numpy as np

from alphago.go.gotypes import Player, Point
from alphago.go.goboard import Move
from .base import Encoder
from alphago.agent.helpers import is_point_an_eye


PLANE_COLOR = slice(0, 3)  # [0–3)   stone color planes
PLANE_ONES = 3  # [3]
PLANE_ZEROS = 4  # [4]
PLANE_SENSIBLE = 5  # [5]

PLANE_TURNS_SINCE = slice(6, 14)  # [6–14)  8 planes
PLANE_LIBERTIES = slice(14, 22)  # [14–22)
PLANE_LIBERTIES_AFTER = slice(22, 30)  # [22–30)
PLANE_CAPTURE_SIZE = slice(30, 38)  # [30–38)
PLANE_SELF_ATARI = slice(38, 46)  # [38–46)

PLANE_LADDER_CAPTURE = 46  # [46]
PLANE_LADDER_ESCAPE = 47  # [47]
PLANE_PLAYER_COLOR = 48  # [48] optional


class AlphaGoEncoder(Encoder):
    """
    Plane layout:
        0-2   : (Stone color) current player, opponent, empty
        3     : Ones
        4     : Zeros
        5     : (Sensibleness) legal moves that don't fill the player's own eyes
        6-13  : (Turns since) binary planes for 1..8+ moves ago
        14-21 : (Liberties) liberty count of the group at each point (1..8+)
        22-29 : (Liberties after move) liberties the new group would have
        30-37 : (Capture size) opponent stones captured by this move
        38-45 : (Self-atari size) own stones left with 1 liberty after this move
        46    : (Ladder capture) opponent stone capturable in a ladder
        47    : (Ladder escape) own stone can escape a ladder
        48    : Current player color (only if use_player_plane=True)
    """

    NUM_8_PLANES = 8

    def __init__(self, board_size, use_player_plane=False) -> None:
        if isinstance(board_size, tuple):
            assert board_size[0] == board_size[1], "Only square boards supported"
            board_size = board_size[0]
        self.board_size = board_size
        self.use_player_plane = use_player_plane
        self.num_planes = (
            PLANE_PLAYER_COLOR + 1 if use_player_plane else PLANE_PLAYER_COLOR
        )

    def name(self):
        return "alphago"

    def shape(self):
        return self.num_planes, self.board_size, self.board_size

    def num_points(self):
        return self.board_size * self.board_size

    def encode_point(self, point):
        return (point.row - 1) * self.board_size + (point.col - 1)

    def decode_point_index(self, index):
        return Point(row=index // self.board_size + 1, col=index % self.board_size + 1)

    def encode(self, game_state):
        size = self.board_size
        planes = np.zeros((self.num_planes, size, size), dtype=np.float32)
        board = game_state.board
        neighbors = board._neighbors
        player = game_state.next_player
        opponent = player.other
        is_over = game_state.is_over()

        # planes[3] is filled with 1
        planes[PLANE_ONES] = 1

        go_strings = {}
        for p in neighbors:
            gs = board.get_go_string(p)
            if gs is not None:
                go_strings[p] = gs

        liberties = np.zeros((size, size), dtype=np.int32)
        libs_after = np.zeros((size, size), dtype=np.int32)
        capture_size = np.zeros((size, size), dtype=np.int32)
        self_atari = np.zeros((size, size), dtype=np.int32)

        # stone color, liberties, and the per-move planes for empty points
        for point, point_neighbors in neighbors.items():
            r, c = point.row - 1, point.col - 1
            go_str = go_strings.get(point)

            if go_str is not None:
                if go_str.color == player:
                    planes[PLANE_COLOR.start, r, c] = 1
                elif go_str.color == opponent:
                    planes[PLANE_COLOR.start + 1, r, c] = 1
                liberties[r, c] = go_str.num_liberties
                continue

            planes[PLANE_COLOR.start + 2, r, c] = 1
            if is_over or not game_state._is_legal_play(point):
                continue

            if not is_point_an_eye(board, point, player):
                planes[PLANE_SENSIBLE, r, c] = 1

            libs, captured, own_size = self._move_features(
                point, point_neighbors, player, go_strings
            )
            libs_after[r, c] = libs
            capture_size[r, c] = captured
            if libs == 1:
                self_atari[r, c] = own_size

        self._set_8_planes(planes, PLANE_LIBERTIES.start, liberties)
        self._set_8_planes(planes, PLANE_LIBERTIES_AFTER.start, libs_after)
        self._set_8_planes(planes, PLANE_CAPTURE_SIZE.start, capture_size)
        self._set_8_planes(planes, PLANE_SELF_ATARI.start, self_atari)

        # turns since (6-13)
        # walk backward through game history and record when each point was last played
        move_age = self._calculate_move_ages(game_state, go_strings)
        for age in range(1, 9):
            planes[PLANE_TURNS_SINCE.start + age - 1][move_age == age] = 1

        ladder_capture_results = {}
        ladder_escape_results = {}

        for p, go_str in go_strings.items():
            color = go_str.color
            sid = id(go_str)
            r, c = p.row - 1, p.col - 1

            if color == opponent and go_str.num_liberties == 2:
                if sid not in ladder_capture_results:
                    ladder_capture_results[sid] = self._ladder_captured(
                        game_state, p, player, depth=20
                    )
                if ladder_capture_results[sid]:
                    planes[PLANE_LADDER_CAPTURE, r, c] = 1

            elif color == player and go_str.num_liberties == 1:
                if sid not in ladder_escape_results:
                    ladder_escape_results[sid] = self._ladder_captured(
                        game_state, p, opponent, depth=20
                    )
                if not ladder_escape_results[sid]:
                    planes[PLANE_LADDER_ESCAPE, r, c] = 1

        if self.use_player_plane and player == Player.black:
            planes[48] = 1

        return planes

    @staticmethod
    def _move_features(point, point_neighbors, player, go_strings):
        """
        one pass over the neighbors of an empty point
        returns (liberties after the move, stones captured,
                 size of own group after the move)
        """
        liberties = set()
        our_stones = {point}
        captured = 0
        seen = set()

        for nb in point_neighbors:
            go_str = go_strings.get(nb)
            if go_str is None:
                liberties.add(nb)
            elif go_str.color == player:
                our_stones |= go_str.stones
                liberties |= go_str.liberties
            elif go_str.num_liberties == 1:
                liberties |= go_str.liberties
                liberties |= go_str.stones
                if id(go_str) not in seen:
                    captured += len(go_str.stones)
                    seen.add(id(go_str))

        liberties -= our_stones
        liberties.discard(point)
        return len(liberties), captured, len(our_stones)

    @staticmethod
    def _set_8_planes(planes, base, counts):
        counts = np.minimum(counts, 8)
        for count in range(1, 9):
            planes[base + count - 1] = counts == count

    def _set_8_plane(self, planes, base, r, c, count):
        count = min(max(count, 0), 8)
        if count > 0:
            planes[base + count - 1, r, c] = 1

    def _calculate_move_ages(self, game_state, go_strings):
        ages = np.full((self.board_size, self.board_size), -1, np.int32)
        state = game_state
        for age in range(1, 9):
            mv = state.last_move
            if mv and mv.is_play:
                r, c = mv.point.row - 1, mv.point.col - 1
                if ages[r, c] == -1:
                    ages[r, c] = age
            if state.previous_state is None:
                break
            state = state.previous_state

        # any stone not visited in the last 8 moves is "8+ moves ago"
        occupied = np.zeros((self.board_size, self.board_size), dtype=bool)
        for point in go_strings:
            occupied[point.row - 1, point.col - 1] = True
        ages[(ages == -1) & occupied] = 8
        return ages

    def _ladder_captured(self, game_state, target_point, chaser, depth):
        if depth == 0:
            return False
        go_str = game_state.board.get_go_string(target_point)
        if go_str is None:
            return True
        liberties = list(go_str.liberties)
        if len(liberties) == 0:
            return True
        if len(liberties) >= 3:
            return False

        if game_state.next_player == chaser:
            for lib in liberties:
                move = Move.play(lib)
                if game_state.is_valid_move(move):
                    next_state = game_state.apply_move(move)
                    if self._ladder_captured(
                        next_state, target_point, chaser, depth - 1
                    ):
                        return True
            return False
        else:
            for lib in liberties:
                move = Move.play(lib)
                if game_state.is_valid_move(move):
                    next_state = game_state.apply_move(move)
                    if not self._ladder_captured(
                        next_state, target_point, chaser, depth - 1
                    ):
                        return False
            return True
