import copy
import random

import pytest

from alphago.go import zobrist
from alphago.go import GameState, Move, Player, Point


def play_moves(size, moves, start=None):
    state = start or GameState.new_game(size)
    for move in moves:
        move = Move.pass_turn() if move is None else Move.play(Point(*move))
        assert state.is_valid_move(move), f"{move.point} should be legal"
        state = state.apply_move(move)
    return state


def stone_map(state):
    size = state.board.size
    stones = {}
    for r in range(1, size + 1):
        for c in range(1, size + 1):
            color = state.board.get(Point(r, c))
            if color is not None:
                stones[(r, c)] = color
    return stones


def random_game(size, seed, max_moves=250):
    rng = random.Random(seed)
    state = GameState.new_game(size)
    states, snapshots = [state], [stone_map(state)]
    points = [Point(r, c) for r in range(1, size + 1) for c in range(1, size + 1)]
    for _ in range(max_moves):
        rng.shuffle(points)
        move = next(
            (Move.play(p) for p in points if state.is_valid_move(Move.play(p))),
            Move.pass_turn(),
        )
        state = state.apply_move(move)
        states.append(state)
        snapshots.append(stone_map(state))
        if state.is_over():
            break
    return states, snapshots


def reference_is_valid(state, move):
    if state.is_over():
        return False
    if not move.is_play:
        return True
    if state.board.get(move.point) is not None:
        return False
    board = copy.deepcopy(state.board)
    board.place_stone(state.next_player, move.point)
    if board.get_go_string(move.point).num_liberties == 0:
        return False
    history = set()
    node = state.previous_state
    while node is not None:
        history.add((node.next_player, node.board.zobrist_hash()))
        node = node.previous_state
    return (state.next_player.other, board.zobrist_hash()) not in history


class TestCaptures:
    def test_single_stone(self):
        state = play_moves(5, [(1, 2), (1, 1), (2, 1)])
        assert stone_map(state) == {(1, 2): Player.black, (2, 1): Player.black}

    def test_two_stone_group(self):
        state = play_moves(5, [(2, 1), (1, 1), (2, 2), (1, 2), (1, 3)])
        assert (1, 1) not in stone_map(state)
        assert (1, 2) not in stone_map(state)

    def test_two_groups_with_one_move(self):
        state = play_moves(
            5, [(2, 1), (1, 1), (2, 3), (1, 3), (1, 4), None, (1, 2)]
        )
        assert stone_map(state) == {
            (1, 2): Player.black,
            (2, 1): Player.black,
            (2, 3): Player.black,
            (1, 4): Player.black,
        }

    def test_restores_liberties(self):
        state = play_moves(5, [(1, 2), (1, 1), (2, 1)])
        black = state.board.get_go_string(Point(1, 2))
        assert black.liberties == {Point(1, 1), Point(1, 3), Point(2, 2)}


class TestLegality:
    def test_occupied_point(self):
        state = play_moves(5, [(3, 3)])
        assert not state.is_valid_move(Move.play(Point(3, 3)))

    def test_single_stone_suicide(self):
        state = play_moves(5, [(1, 2), None, (2, 1)])
        assert not state.is_valid_move(Move.play(Point(1, 1)))

    def test_group_suicide(self):
        state = play_moves(5, [(2, 1), (1, 1), (2, 2), None, (1, 3)])
        assert state.next_player == Player.white
        assert not state.is_valid_move(Move.play(Point(1, 2)))

    def test_suicide_that_captures_is_legal(self):
        state = play_moves(5, [(1, 3), (1, 2), (2, 2), (2, 1), (3, 1), None])
        state = play_moves(5, [(1, 1)], start=state)
        assert (1, 2) not in stone_map(state)
        assert (2, 1) not in stone_map(state)

    def test_pass_and_resign(self):
        state = GameState.new_game(5)
        assert state.is_valid_move(Move.pass_turn())
        assert state.is_valid_move(Move.resign())

    def test_nothing_legal_after_game_over(self):
        state = play_moves(5, [None, None])
        assert state.is_over()
        assert not state.is_valid_move(Move.pass_turn())
        assert not state.is_valid_move(Move.play(Point(3, 3)))


KO_SETUP = [(2, 2), (1, 4), (1, 3), (3, 4), (3, 3), (2, 5), (2, 4), (2, 3)]


class TestKo:
    def test_white_captures_into_ko(self):
        state = play_moves(6, KO_SETUP)
        assert (2, 4) not in stone_map(state)
        assert stone_map(state)[(2, 3)] == Player.white

    def test_immediate_retake_is_illegal(self):
        state = play_moves(6, KO_SETUP)
        assert not state.is_valid_move(Move.play(Point(2, 4)))

    def test_retake_after_ko_threat_is_legal(self):
        state = play_moves(6, KO_SETUP + [(6, 6), (6, 1)])
        assert state.is_valid_move(Move.play(Point(2, 4)))


def test_snapback():
    setup = [
        (3, 1), (2, 1), (3, 2), (2, 2), (3, 3), (2, 3),
        (3, 4), (1, 3), (2, 4), None, (1, 4), None,
    ]
    state = play_moves(5, setup + [(1, 2), (1, 1)])
    assert (1, 2) not in stone_map(state)
    state = play_moves(5, [(1, 2)], start=state)
    for point in [(1, 1), (1, 3), (2, 1), (2, 2), (2, 3)]:
        assert point not in stone_map(state)
    assert stone_map(state)[(1, 2)] == Player.black


@pytest.mark.parametrize("size,seed", [(5, 1), (7, 2), (9, 3), (9, 4), (13, 5)])
class TestRandomGames:
    def test_board_invariants(self, size, seed):
        for state in random_game(size, seed)[0]:
            board = state.board
            expected_hash = zobrist.EMPTY_BOARD
            for point, string in board._grid.items():
                assert point in string.stones
                expected_hash ^= zobrist.HASH_CODE[point, string.color]
                expected_libs = {
                    nb
                    for stone in string.stones
                    for nb in stone.neighbors()
                    if board.is_on_board(nb) and board.get(nb) is None
                }
                assert set(string.liberties) == expected_libs
                assert string.num_liberties > 0
            assert board.zobrist_hash() == expected_hash

    def test_legal_moves_match_reference(self, size, seed):
        states, _ = random_game(size, seed)
        for state in states[:: max(1, len(states) // 12)]:
            expected = [
                Point(r, c)
                for r in range(1, size + 1)
                for c in range(1, size + 1)
                if reference_is_valid(state, Move.play(Point(r, c)))
            ]
            got = [m.point for m in state.legal_moves() if m.is_play]
            assert got == expected

    def test_history_is_not_mutated(self, size, seed):
        states, snapshots = random_game(size, seed)
        assert [stone_map(s) for s in states] == snapshots
