import time
from dlgo import goboard
from dlgo.goboard import Point, Move
from dlgo.utils import GoBoardDisplay, print_move
from dlgo.data.process_files import read_txt_files, extract_moves, extract_winner


def main():
    board_size = 19
    folder = "dlgo/data/go_games"
    game = goboard.GameState.new_game(board_size)
    games = read_txt_files(folder)
    moves = extract_moves(games[0])

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
