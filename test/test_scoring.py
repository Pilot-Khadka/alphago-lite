from alphago.go import Board, GameState, Move, Player, Point, compute_game_result


def board_with(size, black=(), white=()):
    board = Board(size)
    for point in black:
        board.place_stone(Player.black, Point(*point))
    for point in white:
        board.place_stone(Player.white, Point(*point))
    return board


def column(col, size=5):
    return [(row, col) for row in range(1, size + 1)]


def test_empty_board_is_all_dame():
    result = compute_game_result(Board(5), komi=7.5)
    assert result.dame == 25
    assert result.black_score == 0
    assert result.white_score == 7.5
    assert result.winner == Player.white
    assert str(result) == "W+7.5"


def test_walls_split_the_board():
    board = board_with(5, black=column(2), white=column(4))
    result = compute_game_result(board, komi=7.5)
    assert (result.black_stones, result.white_stones) == (5, 5)
    assert (result.black_territory, result.white_territory) == (5, 5)
    assert result.dame == 5
    assert str(result) == "W+7.5"


def test_equal_score_without_komi_is_a_draw():
    board = board_with(5, black=column(2), white=column(4))
    result = compute_game_result(board, komi=0)
    assert result.winner is None
    assert str(result) == "Draw"


def test_region_touching_one_color_is_territory():
    board = board_with(5, black=column(3))
    result = compute_game_result(board, komi=7.5)
    assert result.black_territory == 20
    assert result.black_score == 25
    assert str(result) == "B+17.5"


def test_eye_counts_and_contested_region_does_not():
    board = board_with(5, black=[(1, 2), (2, 1), (2, 2)], white=[(3, 3)])
    result = compute_game_result(board, komi=0)
    assert result.black_territory == 1
    assert result.white_territory == 0
    assert result.dame == 25 - 4 - 1


def test_two_passes_use_area_score():
    state = GameState.new_game(5, komi=0.5)
    for point in [(1, 3), (1, 4), (2, 3), (2, 4), (3, 3), (3, 4), (4, 3), (4, 4),
                  (5, 3), (5, 4)]:
        state = state.apply_move(Move.play(Point(*point)))
    state = state.apply_move(Move.pass_turn()).apply_move(Move.pass_turn())
    assert state.is_over()
    assert state.result().black_score == 15
    assert state.result().white_score == 10.5
    assert state.winner() == Player.black


def test_resignation_beats_score():
    state = GameState.new_game(5)
    state = state.apply_move(Move.play(Point(3, 3)))
    state = state.apply_move(Move.resign())
    assert state.winner() == Player.black


def test_komi_is_carried_through_moves():
    state = GameState.new_game(9, komi=0.5)
    state = state.apply_move(Move.play(Point(5, 5))).apply_move(Move.pass_turn())
    assert state.komi == 0.5
    assert state.result().komi == 0.5


def test_no_winner_before_game_over():
    state = GameState.new_game(5).apply_move(Move.play(Point(3, 3)))
    assert state.winner() is None
    assert state.result().winner == Player.black
