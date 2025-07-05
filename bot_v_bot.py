from dlgo import gotypes
from dlgo import agent
from dlgo import goboard_slow
from dlgo.utils import print_move, GoBoardDisplay


def main():
    board_size = 19
    game = goboard_slow.GameState.new_game(board_size)
    bots = {
        gotypes.Player.black: agent.naive.RandomBot(),
        gotypes.Player.white: agent.naive.RandomBot(),
    }
    display = GoBoardDisplay(board_size)
    while not game.is_over():
        # time.sleep(0.01)

        display.update_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)


if __name__ == "__main__":
    main()
