import pytest

from alphago.data.games import (
    GameArchive,
    InvalidGame,
    parse_sgf,
    replay,
    save_games,
)
from alphago.go import Player, Point


def test_moves_komi_and_result():
    record = parse_sgf("(;GM[1]SZ[19]KM[6.5]RE[W+3.5];B[pd];W[dp];B[pp])")
    assert record.board_size == 19
    assert record.komi == 6.5
    assert record.winner == Player.white
    assert record.first_player == Player.black
    assert record.moves == [(4, 16), (16, 4), (16, 16)]


def test_text_in_comments_and_names_is_not_a_move():
    sgf = "(;SZ[19]PB[Some;one]C[try B[aa\\] here; W[bb]];B[cc]C[a;b])"
    record = parse_sgf(sgf)
    assert record.moves == [(3, 3)]


def test_passes():
    record = parse_sgf("(;SZ[19];B[aa];W[];B[tt];W[bb])")
    assert record.moves == [(1, 1), None, None, (2, 2)]


def test_handicap_setup_and_white_first():
    record = parse_sgf("(;SZ[19]HA[2]KM[0.5]AB[dd][pp];W[dp];B[pd])")
    assert record.black_setup == [(4, 4), (16, 16)]
    assert record.first_player == Player.white
    states = [s for s, _ in replay(record)]
    assert states[0].next_player == Player.white
    assert states[0].board.get(Point(4, 4)) == Player.black
    assert states[0].komi == 0.5


def test_same_color_twice_is_rejected():
    with pytest.raises(InvalidGame):
        parse_sgf("(;SZ[19];B[aa];B[bb])")


def test_coordinate_outside_board_is_rejected():
    with pytest.raises(InvalidGame):
        parse_sgf("(;SZ[9];B[kk])")


def test_replay_rejects_illegal_move():
    record = parse_sgf("(;SZ[19];B[aa];W[aa])")
    with pytest.raises(InvalidGame):
        list(replay(record))


def test_replay_alternates_players():
    record = parse_sgf("(;SZ[9];B[ee];W[cc];B[];W[gg])")
    players = [state.next_player for state, _ in replay(record)]
    assert players == [Player.black, Player.white, Player.black, Player.white]


def test_save_and_load_roundtrip(tmp_path):
    records = [
        parse_sgf("(;SZ[19]KM[7.5]RE[B+R];B[pd];W[];B[dd])"),
        parse_sgf("(;SZ[19]KM[0.5]AB[dd][pp]RE[W+2];W[dp];B[pd])"),
    ]
    path = tmp_path / "games.npz"
    save_games(path, records)
    archive = GameArchive(path)
    assert len(archive) == 2
    assert archive.num_moves() == 5
    for i, record in enumerate(records):
        loaded = archive[i]
        assert loaded.moves == record.moves
        assert loaded.black_setup == record.black_setup
        assert loaded.first_player == record.first_player
        assert loaded.komi == record.komi
        assert loaded.winner == record.winner
        assert loaded.key() == record.key()
