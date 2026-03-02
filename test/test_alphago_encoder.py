import pytest
from unittest.mock import patch, MagicMock

import numpy as np

from alphago.go.goboard import GoString
from alphago.go import GameState, Board, Move, Player, Point
from alphago.encoders.alphago_encoder import AlphaGoEncoder


BOARD_SIZE = 19
ENC = AlphaGoEncoder(board_size=BOARD_SIZE)


def make_board(stone_map=None):
    board = Board(BOARD_SIZE)
    if stone_map:
        for p, player in stone_map.items():
            board.place_stone(player, p)
    return board


def make_game_state(
    board=None, next_player=Player.black, last_move=None, previous_state=None
):
    if board is None:
        board = Board(BOARD_SIZE)
    return GameState(
        board=board,
        next_player=next_player,
        previous_state=previous_state,
        last_move=last_move,
    )


def empty_game_state(next_player=Player.black):
    board = Board(BOARD_SIZE)
    return GameState(
        board=board,
        next_player=next_player,
        previous_state=None,
        last_move=None,
    )


def make_play(point):
    return Move.play(point)


class TestConstructor:
    def test_scalar_board_size(self):
        enc = AlphaGoEncoder(board_size=9)
        assert enc.board_size == 9
        assert enc.num_planes == 48

    def test_tuple_board_size(self):
        enc = AlphaGoEncoder(board_size=(9, 9))
        assert enc.board_size == 9

    def test_non_square_tuple_raises(self):
        with pytest.raises(AssertionError):
            AlphaGoEncoder(board_size=(9, 13))

    def test_player_plane_adds_one(self):
        enc = AlphaGoEncoder(board_size=9, use_player_plane=True)
        assert enc.num_planes == 49

    def test_shape(self):
        assert ENC.shape() == (48, BOARD_SIZE, BOARD_SIZE)

    def test_num_points(self):
        assert ENC.num_points() == BOARD_SIZE**2

    def test_name(self):
        assert ENC.name() == "alphago"


class TestPointEncoding:
    def test_encode_point_origin(self):
        assert ENC.encode_point(Point(1, 1)) == 0

    def test_encode_decode_roundtrip(self):
        for r in range(1, BOARD_SIZE + 1):
            for c in range(1, BOARD_SIZE + 1):
                pt = Point(r, c)
                assert ENC.decode_point_index(ENC.encode_point(pt)) == pt


class TestPlane0_CurrentPlayer:
    def test_current_player_stone_set(self):
        pt = Point(1, 1)
        board = make_board(stone_map={pt: Player.black})
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        assert planes[0, 0, 0] == 1
        assert planes[1, 0, 0] == 0
        assert planes[2, 0, 0] == 0

    def test_opponent_stone_on_plane_1(self):
        pt = Point(1, 1)
        board = make_board(stone_map={pt: Player.white})
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        assert planes[0, 0, 0] == 0
        assert planes[1, 0, 0] == 1

    def test_empty_point_on_plane_2(self):
        board = make_board()
        state = make_game_state(board)
        planes = ENC.encode(state)
        assert planes[2].sum() == BOARD_SIZE**2


class TestPlanes3And4:
    def test_plane_3_all_ones(self):
        planes = ENC.encode(empty_game_state())
        assert (planes[3] == 1).all()

    def test_plane_4_all_zeros(self):
        planes = ENC.encode(empty_game_state())
        assert (planes[4] == 0).all()


class TestPlane5_Sensibleness:
    def _state_with_surrounded_empty(self):
        eye = Point(3, 3)
        surrounding = {
            Point(2, 3): Player.black,
            Point(4, 3): Player.black,
            Point(3, 2): Player.black,
            Point(3, 4): Player.black,
            Point(2, 2): Player.black,
            Point(2, 4): Player.black,
            Point(4, 2): Player.black,
            Point(4, 4): Player.black,
        }
        board = make_board(stone_map=surrounding)
        return make_game_state(board, next_player=Player.black), eye

    def test_eye_excluded_from_sensibleness(self):
        state, eye = self._state_with_surrounded_empty()
        planes = ENC.encode(state)
        assert planes[5, eye.row - 1, eye.col - 1] == 0

    def test_normal_empty_point_is_sensible(self):
        board = make_board()
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        assert planes[5, 0, 0] == 1

    def test_occupied_point_not_sensible(self):
        pt = Point(1, 1)
        board = make_board(stone_map={pt: Player.black})
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        assert planes[5, 0, 0] == 0

    def test_all_occupied_board_has_no_sensible_moves(self):
        # Fill every point so no empty cells exist → plane 5 all zeros
        stone_map = {}
        for r in range(1, BOARD_SIZE + 1):
            for c in range(1, BOARD_SIZE + 1):
                stone_map[Point(r, c)] = (
                    Player.black if (r + c) % 2 == 0 else Player.white
                )
        board = make_board(stone_map=stone_map)
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        assert planes[5].sum() == 0


class TestPlanes6to13_TurnsSince:
    def test_most_recent_move_on_plane_6(self):
        pt = Point(1, 1)
        board = make_board()
        state = make_game_state(board, last_move=make_play(pt), previous_state=None)
        planes = ENC.encode(state)
        assert planes[6, 0, 0] == 1

    def test_two_moves_ago_on_plane_7(self):
        pt1 = Point(1, 1)
        pt2 = Point(2, 2)
        board = make_board()

        prev_state = make_game_state(
            board, last_move=make_play(pt2), previous_state=None
        )
        curr_state = make_game_state(
            board, last_move=make_play(pt1), previous_state=prev_state
        )

        planes = ENC.encode(curr_state)
        assert planes[6, 0, 0] == 1  # pt1 is 1 move ago -> plane 6
        assert planes[7, 1, 1] == 1  # pt2 is 2 moves ago -> plane 7

    def test_pass_move_does_not_set_plane(self):
        board = make_board()
        state = make_game_state(board, last_move=Move.pass_turn(), previous_state=None)
        planes = ENC.encode(state)
        assert planes[6:14].sum() == 0

    def test_eight_plus_moves_clamped_to_plane_13(self):
        board = make_board()

        def make_state_chain(n):
            state = None
            for i in range(1, n + 1):
                pt = Point(i + 1, 1)
                state = make_game_state(
                    board, last_move=make_play(pt), previous_state=state
                )
            return state

        root = make_game_state(
            board,
            last_move=make_play(Point(1, 1)),
            previous_state=make_state_chain(7),
        )
        planes = ENC.encode(root)
        assert planes[13, 1, 0] == 1  # Point(2,1) at age 8 -> plane 13


class TestPlanes14to21_Liberties:
    """
    Board layouts for specific liberty counts, all checking a stone at (3,3):
      - 1 liberty:  (3,3) black + white blocking 3 of 4 neighbors
      - 2 liberties: (3,3) black + white blocking 2 neighbors
      - 4 liberties: (3,3) black alone (interior point)
      - 8 liberties: 3-stone chain (3,2)-(3,3)-(3,4) has 8 shared liberties
      - 9 liberties: 4-stone chain (3,1)-(3,4) has 9 shared liberties
    """

    def _state_1_liberty(self):
        board = make_board(
            stone_map={
                Point(3, 3): Player.black,
                Point(2, 3): Player.white,
                Point(4, 3): Player.white,
                Point(3, 2): Player.white,
            }
        )
        return make_game_state(board, next_player=Player.black)

    def _state_2_liberties(self):
        board = make_board(
            stone_map={
                Point(3, 3): Player.black,
                Point(2, 3): Player.white,
                Point(4, 3): Player.white,
            }
        )
        return make_game_state(board, next_player=Player.black)

    def _state_4_liberties(self):
        board = make_board(stone_map={Point(3, 3): Player.black})
        return make_game_state(board, next_player=Player.black)

    def _state_8_liberties(self):
        # Chain (3,2)-(3,3)-(3,4): liberties are (2,2),(4,2),(3,1),(2,3),(4,3),(2,4),(4,4),(3,5) = 8
        board = make_board(
            stone_map={
                Point(3, 2): Player.black,
                Point(3, 3): Player.black,
                Point(3, 4): Player.black,
            }
        )
        return make_game_state(board, next_player=Player.black)

    def _state_9_liberties(self):
        # Chain (3,1)-(3,4): adds (2,1),(4,1) while (3,0) is off-board → 9 liberties
        board = make_board(
            stone_map={
                Point(3, 1): Player.black,
                Point(3, 2): Player.black,
                Point(3, 3): Player.black,
                Point(3, 4): Player.black,
            }
        )
        return make_game_state(board, next_player=Player.black)

    @pytest.mark.parametrize(
        "state_fn,check_point,expected_plane",
        [
            ("_state_1_liberty", Point(3, 3), 14),
            ("_state_2_liberties", Point(3, 3), 15),
            ("_state_4_liberties", Point(3, 3), 17),
            ("_state_8_liberties", Point(3, 3), 21),
            ("_state_9_liberties", Point(3, 3), 21),
        ],
    )
    def test_liberty_plane_selection(self, state_fn, check_point, expected_plane):
        state = getattr(self, state_fn)()
        planes = ENC.encode(state)
        r, c = check_point.row - 1, check_point.col - 1
        assert planes[expected_plane, r, c] == 1

    def test_empty_point_has_no_liberty_plane_set(self):
        board = make_board()
        state = make_game_state(board)
        planes = ENC.encode(state)
        assert planes[14:22].sum() == 0


class TestPlanes22to29_LibertiesAfterMove:
    def test_empty_board_corner_move_has_two_liberties(self):
        board = make_board()
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        assert planes[22 + 2 - 1, 0, 0] == 1  # corner → 2 libs → plane 23

    def test_empty_board_center_move_has_four_liberties(self):
        board = make_board()
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        assert planes[22 + 4 - 1, 2, 2] == 1  # center → 4 libs → plane 25


class TestPlanes30to37_CaptureSize:
    def test_capturing_one_stone(self):
        """
        White at (1,2) surrounded by black at (1,3) and (2,2), leaving only
        liberty (1,1). Black playing (1,1) captures exactly 1 white stone.
        """
        board = make_board(
            stone_map={
                Point(1, 2): Player.white,
                Point(1, 3): Player.black,
                Point(2, 2): Player.black,
            }
        )
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        assert planes[30, 0, 0] == 1  # capture_size=1 -> plane 30

    def test_no_capture_no_plane_set(self):
        board = make_board()
        state = make_game_state(board)
        planes = ENC.encode(state)
        assert planes[30:38].sum() == 0


class TestPlanes38to45_SelfAtari:
    def test_no_self_atari_on_open_board(self):
        board = make_board()
        state = make_game_state(board)
        planes = ENC.encode(state)
        assert planes[38:46].sum() == 0

    def test_self_atari_detected(self):
        """
        Black at (1,2) whose only open liberty is (1,1), because (1,3)
        and (2,2) are white. Playing black at (1,1): the new group
        {(1,1),(1,2)} has a single liberty at (2,1) → self-atari.
        """
        board = make_board(
            stone_map={
                Point(1, 2): Player.black,
                Point(1, 3): Player.white,
                Point(2, 2): Player.white,
            }
        )
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        # After playing (1,1) the group has 1 liberty → self-atari, size=1+1=2
        assert planes[38, 0, 0] == 1 or planes[39, 0, 0] == 1


class TestPlane46_LadderCapture:
    def test_capturable_opponent_in_ladder_sets_plane(self):
        pt = Point(2, 2)

        board = make_board(stone_map={pt: Player.white})
        state = make_game_state(board, next_player=Player.black)

        # Force the go string to have exactly 2 liberties
        go_str_mock = MagicMock()
        go_str_mock.color = Player.white
        go_str_mock.num_liberties = 2
        go_str_mock.stones = {pt}
        go_str_mock.liberties = {Point(1, 1), Point(1, 2)}

        with patch.object(state.board, "get_go_string", return_value=go_str_mock):
            with patch(
                "alphago.encoders.alphago_encoder.AlphaGoEncoder._ladder_captured",
                return_value=True,
            ):
                planes = ENC.encode(state)

        # Now ladder capture plane should be set
        assert planes[46, 1, 1] == 1

    def test_not_capturable_does_not_set_plane(self):
        pt = Point(2, 2)
        board = make_board(stone_map={pt: Player.white})
        state = make_game_state(board, next_player=Player.black)

        with patch.object(ENC, "_ladder_captured", return_value=False):
            planes = ENC.encode(state)

        assert planes[46, 1, 1] == 0

    def test_group_with_3_liberties_not_checked(self):
        """
        White at (3,3) alone has 4 liberties; the encoder only
        checks ladder capture for groups with exactly 2 liberties.
        """
        pt = Point(3, 3)
        board = make_board(stone_map={pt: Player.white})
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        assert planes[46, 2, 2] == 0


class TestPlane47_LadderEscape:
    def test_own_stone_can_escape_sets_plane(self):
        """Black in atari (1 liberty) that the ladder search says can escape."""
        board = make_board(
            stone_map={
                Point(2, 2): Player.black,
                Point(1, 2): Player.white,
                Point(2, 1): Player.white,
                Point(2, 3): Player.white,
            }
        )
        state = make_game_state(board, next_player=Player.black)

        with patch.object(ENC, "_ladder_captured", return_value=False):
            planes = ENC.encode(state)

        assert planes[47, 1, 1] == 1

    def test_own_stone_cannot_escape_does_not_set_plane(self):
        board = make_board(
            stone_map={
                Point(2, 2): Player.black,
                Point(1, 2): Player.white,
                Point(2, 1): Player.white,
                Point(2, 3): Player.white,
            }
        )
        state = make_game_state(board, next_player=Player.black)

        with patch.object(ENC, "_ladder_captured", return_value=True):
            planes = ENC.encode(state)

        assert planes[47, 1, 1] == 0

    def test_own_stone_with_2_liberties_not_checked(self):
        pt = Point(2, 2)
        board = make_board(stone_map={pt: Player.black})
        state = make_game_state(board, next_player=Player.black)
        planes = ENC.encode(state)
        assert planes[47, 1, 1] == 0


class TestPlane48_PlayerPlane:
    def test_black_to_play_fills_plane_48(self):
        enc = AlphaGoEncoder(board_size=BOARD_SIZE, use_player_plane=True)
        state = empty_game_state(next_player=Player.black)
        planes = enc.encode(state)
        assert (planes[48] == 1).all()

    def test_white_to_play_leaves_plane_48_zero(self):
        enc = AlphaGoEncoder(board_size=BOARD_SIZE, use_player_plane=True)
        state = empty_game_state(next_player=Player.white)
        planes = enc.encode(state)
        assert (planes[48] == 0).all()

    def test_player_plane_absent_without_flag(self):
        enc = AlphaGoEncoder(board_size=BOARD_SIZE, use_player_plane=False)
        state = empty_game_state(next_player=Player.black)
        planes = enc.encode(state)
        assert planes.shape[0] == 48


class TestOutputProperties:
    def test_output_shape(self):
        planes = ENC.encode(empty_game_state())
        assert planes.shape == (48, BOARD_SIZE, BOARD_SIZE)

    def test_output_dtype_float32(self):
        planes = ENC.encode(empty_game_state())
        assert planes.dtype == np.float32

    def test_planes_are_binary(self):
        planes = ENC.encode(empty_game_state())
        assert set(np.unique(planes)).issubset({0.0, 1.0})


class TestSetPlane8:
    def setup_method(self):
        self.enc = AlphaGoEncoder(board_size=1)

    def test_count_clamped_to_8(self):
        planes = np.zeros((22 + 8, 1, 1), dtype=np.float32)
        self.enc._set_8_plane(planes, 14, 0, 0, 100)
        # base + 8 - 1 = 14 + 7 = 21
        assert planes[21, 0, 0] == 1

    def test_count_1_sets_base(self):
        planes = np.zeros((22 + 8, 1, 1), dtype=np.float32)
        self.enc._set_8_plane(planes, 14, 0, 0, 1)
        assert planes[14, 0, 0] == 1


class TestLadderCaptured:
    def test_zero_liberties_is_captured(self):
        """
        A GoString with 0 liberties cannot exist in real play but we construct
        one directly to test the base-case branch of _ladder_captured.
        """
        # Make a fake single-stone group with 0 liberties
        gs = GoString(Player.white, frozenset({Point(1, 1)}), frozenset())

        board = Board(BOARD_SIZE)
        board._grid[Point(1, 1)] = gs
        state = make_game_state(board, next_player=Player.black)
        result = ENC._ladder_captured(state, Point(1, 1), Player.black, depth=5)
        assert result is True

    def test_three_liberties_escapes(self):
        """A group with 3+ liberties always escapes; no further search needed."""
        board = make_board(stone_map={Point(3, 3): Player.white})
        state = make_game_state(board, next_player=Player.black)
        result = ENC._ladder_captured(state, Point(3, 3), Player.black, depth=5)
        assert result is False

    def test_depth_zero_returns_false(self):
        """Exhausted search depth → conservatively treat as non-captured."""
        board = make_board(
            stone_map={
                Point(3, 3): Player.white,
                Point(2, 3): Player.black,
                Point(3, 2): Player.black,
            }
        )
        state = make_game_state(board, next_player=Player.black)
        result = ENC._ladder_captured(state, Point(3, 3), Player.black, depth=0)
        assert result is False
