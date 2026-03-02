import time

from alphago.go import goboard
from alphago.go.goboard import Point, Move
from alphago.util.display_board import GoBoardDisplay, print_move
from alphago.data import (
    read_txt_files,
    extract_moves,
    extract_winner,
    setup_handicap_game,
)


def main():
    board_size = 19
    folder = "external/computer-go-dataset/Professional/pro2000+.txt"
    game = goboard.GameState.new_game(board_size)
    games = read_txt_files(folder)
    moves, handicap_info = extract_moves(games[0])

    if handicap_info["is_handicap_game"]:
        # setup game with handicap
        game, handicap_moves = setup_handicap_game(board_size, handicap_info)

    display = GoBoardDisplay(board_size)
    for move in moves:
        color, coords = move
        if coords is None:
            continue
        row, col = coords

        point = Point(row, col)

        move_obj = Move.play(point)

        display.update_board(game.board)
        print_move(game.next_player, move_obj)
        game = game.apply_move(move_obj)
        time.sleep(0.02)

    winner = extract_winner(games[0])
    print(winner)


if __name__ == "__main__":
    main()
