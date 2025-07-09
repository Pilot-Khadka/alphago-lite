import time
from dlgo import goboard
from dlgo.goboard import Point, Move
from dlgo.utils import GoBoardDisplay, print_move
from dlgo.data.process_files import (
    read_txt_files,
    extract_moves,
    extract_winner,
)


def get_standard_handicap_positions(board_size, handicap_count):
    if board_size == 19:
        star_points = [
            (3, 3),  # 0: top-left
            (3, 9),  # 1: top-center
            (3, 15),  # 2: top-right
            (9, 3),  # 3: center-left
            (9, 9),  # 4: center
            (9, 15),  # 5: center-right
            (15, 3),  # 6: bottom-left
            (15, 9),  # 7: bottom-center
            (15, 15),  # 8: bottom-right
        ]
    else:
        # only supports 19x19 for now
        raise NotImplementedError()

    # based on handicap count, return num. of appropriate stones
    handicap_sequences = {
        2: [2, 6],  # Upper right, lower left
        3: [2, 6, 8],  # Add lower right
        4: [2, 6, 0, 8],  # Add top left
        5: [2, 6, 0, 8, 4],  # + center
        6: [2, 6, 0, 8, 1, 7],  # + top/bottom center
        7: [2, 6, 0, 8, 1, 7, 4],
        8: [2, 6, 0, 8, 1, 3, 5, 7],
        9: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }

    if handicap_count in handicap_sequences:
        indices = handicap_sequences[handicap_count]
        return [star_points[i] for i in indices]

    return []


def setup_handicap_game(board_size, handicap_info):
    game = goboard.GameState.new_game(board_size)
    handicap_moves = []

    if not handicap_info["is_handicap_game"]:
        return game, handicap_moves

    # if position in SGF is available, use that
    if handicap_info["handicap_stones"]:
        positions = handicap_info["handicap_stones"]
    else:
        positions = get_standard_handicap_positions(
            board_size, handicap_info["handicap_count"]
        )

    # place handicap stones (always black)
    for row, col in positions:
        point = Point(row, col)
        move = Move.play(point)

        if game.is_valid_move(move):
            game = game.apply_move(move)
            handicap_moves.append(("black", (row, col)))
        else:
            print(f"Warning: Invalid handicap stone at {point}")

    return game, handicap_moves


def main():
    board_size = 19
    folder = "dlgo/data/go_games"
    game = goboard.GameState.new_game(board_size)
    games = read_txt_files(folder)
    moves, handicap_info = extract_moves(games[0])

    if handicap_info["is_handicap_game"]:
        # setup game with handicap
        game, handicap_moves = setup_handicap_game(board_size, handicap_info)

    display = GoBoardDisplay(board_size)
    for move in moves:
        print("move:", move)
        color, (row, col) = move

        point = Point(row, col)

        move_obj = Move.play(point)

        display.update_board(game.board)
        print_move(game.next_player, move_obj)
        game = game.apply_move(move_obj)
        time.sleep(0.2)

    winner = extract_winner(games[0])
    print(winner)


if __name__ == "__main__":
    main()
