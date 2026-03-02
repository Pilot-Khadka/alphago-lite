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
        planes = np.zeros(
            (self.num_planes, self.board_size, self.board_size),
            dtype=np.float32,
        )
        board = game_state.board
        player = game_state.next_player
        opponent = player.other

        # planes[3] is filled with 1
        planes[PLANE_ONES] = 1

        # stone color and liberties planes
        for r in range(self.board_size):
            for c in range(self.board_size):
                point = Point(r + 1, c + 1)
                go_str = board.get_go_string(point)

                if go_str:
                    if go_str.color == player:
                        planes[PLANE_COLOR.start, r, c] = 1
                    elif go_str.color == opponent:
                        planes[PLANE_COLOR.start + 1, r, c] = 1
                else:
                    planes[PLANE_COLOR.start + 2, r, c] = 1

                # liberties (1–8+)
                if go_str:
                    self._set_8_plane(
                        planes, PLANE_LIBERTIES.start, r, c, go_str.num_liberties
                    )

        # turns since (6-13)
        # walk backward through game history and record when each point was last played
        move_age = self._calculate_move_ages(game_state)
        for age in range(1, 9):
            planes[PLANE_TURNS_SINCE.start + age - 1][move_age == age] = 1

        # per move dynamic features
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = Point(r + 1, c + 1)

                if board.get(p) is not None:
                    continue

                move = Move.play(p)
                if not game_state.is_valid_move(move):
                    continue

                # Sensibleness
                if not is_point_an_eye(board, p, player):
                    planes[PLANE_SENSIBLE, r, c] = 1

                # calculate derived features
                libs_after = self._liberties_after_move(board, p, player)
                cap_size = self._capture_size(board, p, player)
                self_atari = self._self_atari_size(board, p, player, libs_after)

                self._set_8_plane(planes, PLANE_LIBERTIES_AFTER.start, r, c, libs_after)
                self._set_8_plane(planes, PLANE_CAPTURE_SIZE.start, r, c, cap_size)
                self._set_8_plane(planes, PLANE_SELF_ATARI.start, r, c, self_atari)

        # ladder features
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = Point(r + 1, c + 1)
                go_str = board.get_go_string(p)
                if go_str is None:
                    continue

                color = board.get(p)
                if color == opponent and go_str.num_liberties == 2:
                    if self._ladder_captured(game_state, p, player, depth=20):
                        planes[PLANE_LADDER_CAPTURE, r, c] = 1

                if color == player and go_str.num_liberties == 1:
                    if not self._ladder_captured(game_state, p, opponent, depth=20):
                        planes[PLANE_LADDER_ESCAPE, r, c] = 1

        # current player color (48)
        if self.use_player_plane and player == Player.black:
            planes[48] = 1

        return planes

    def _set_8_plane(self, planes, base, r, c, count):
        count = min(max(count, 0), 8)
        if count > 0:
            planes[base + count - 1, r, c] = 1

    def _calculate_move_ages(self, game_state):
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
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (
                    ages[r, c] == -1
                    and game_state.board.get(Point(r + 1, c + 1)) is not None
                ):
                    ages[r, c] = 8
        return ages

    def _liberties_after_move(self, board, point, player):
        liberties = set()
        our_stones = {point}
        opponent = player.other

        for nb in point.neighbors():
            if not board.is_on_board(nb):
                continue
            color = board.get(nb)
            if color is None:
                liberties.add(nb)
            elif color == player:
                go_str = board.get_go_string(nb)
                if go_str is None:
                    continue
                our_stones |= go_str.stones
                liberties |= go_str.liberties
            elif color == opponent:
                go_str = board.get_go_string(nb)
                if go_str is None:
                    continue
                if go_str.num_liberties == 1:
                    # freed positions become potential liberties
                    liberties |= go_str.liberties
                    liberties |= go_str.stones

        liberties -= our_stones
        liberties.discard(point)
        return len(liberties)

    def _capture_size(self, board, point, player):
        opponent = player.other
        captured = 0
        seen = set()
        for nb in point.neighbors():
            if not board.is_on_board(nb):
                continue
            if board.get(nb) == opponent:
                go_str = board.get_go_string(nb)
                if go_str is None:
                    continue

                sid = id(go_str)
                if sid not in seen and go_str.num_liberties == 1:
                    captured += len(go_str.stones)
                    seen.add(sid)
        return captured

    def _self_atari_size(self, board, point, player, libs_after=None):
        """Own stones that would be in atari (1 liberty) after placing at point."""
        if libs_after is None:
            libs_after = self._liberties_after_move(board, point, player)
        if libs_after != 1:
            return 0
        size = 1  # the new stone itself
        seen = set()
        for nb in point.neighbors():
            if not board.is_on_board(nb):
                continue
            if board.get(nb) == player:
                go_str = board.get_go_string(nb)
                if go_str is None:
                    continue

                sid = id(go_str)
                if sid not in seen:
                    size += len(go_str.stones)
                    seen.add(sid)
        return size

    def _ladder_captured(self, game_state, target_point, chaser, depth):
        if depth == 0:
            return False
        go_str = game_state.board.get_go_string(target_point)
        if go_str is None:
            return True  # already captured
        liberties = list(go_str.liberties)
        if len(liberties) == 0:
            return True
        if len(liberties) >= 3:
            return False  # escaped

        if game_state.next_player == chaser:
            # Chaser plays: succeeds if any move leads to capture
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
            # Escapee plays: escapes if any move avoids capture
            for lib in liberties:
                move = Move.play(lib)
                if game_state.is_valid_move(move):
                    next_state = game_state.apply_move(move)
                    if not self._ladder_captured(
                        next_state, target_point, chaser, depth - 1
                    ):
                        return False
            return True  # no escape found
