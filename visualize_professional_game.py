import sys
import time
from PyQt6.QtWidgets import QApplication
from dlgo import goboard
from dlgo.goboard import Point, Move

from dlgo.frontend.qt6 import GoBoardApplication
from dlgo.data.process_files import (
    read_txt_files,
    extract_moves,
    extract_winner,
    setup_handicap_game,
)


def main():
    app = QApplication(sys.argv)

    board_size = 19
    folder = "go_data/go_games"
    game = goboard.GameState.new_game(board_size)
    games = read_txt_files(folder)
    moves, handicap_info = extract_moves(games[0])

    if handicap_info["is_handicap_game"]:
        game, handicap_moves = setup_handicap_game(board_size, handicap_info)

    display = GoBoardApplication(board_size)
    display.show()

    for move in moves:
        print("move:", move)
        color, (row, col) = move
        point = Point(row, col)
        move_obj = Move.play(point)
        display.board_widget.update_board(game.board)
        game = game.apply_move(move_obj)

        app.processEvents()
        time.sleep(0.2)

    winner = extract_winner(games[0])
    print(winner)

    print("Game finished. Close the window to exit.")
    app.exec()


if __name__ == "__main__":
    main()
