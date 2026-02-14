import time
import statistics

from alphago import gotypes
from alphago.agent.mcts import MCTSBot
from alphago import goboard
from alphago.display_board import print_move, GoBoardDisplay


def main():
    board_size = 19
    game = goboard.GameState.new_game(board_size)
    bots = {
        gotypes.Player.black: MCTSBot(),
        gotypes.Player.white: MCTSBot(),
    }
    display = GoBoardDisplay(board_size)

    start_time = time.time()
    move_times = []
    total_moves = 0

    print(f"Starting Go game with board size {board_size}x{board_size}")
    print("-" * 50)

    while not game.is_over():
        move_start = time.time()

        display.update_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)

        move_end = time.time()
        move_time = move_end - move_start
        move_times.append(move_time)
        total_moves += 1

        if total_moves % 10 == 0:
            avg_time = statistics.mean(move_times[-10:])
            print(f"  [Move {total_moves:3d}] Last 10 moves avg: {avg_time:.4f}s")

    total_time = time.time() - start_time
    avg_move_time = statistics.mean(move_times)
    median_move_time = statistics.median(move_times)
    max_move_time = max(move_times)
    min_move_time = min(move_times)

    print(f"Total game time:      {total_time:.2f} seconds")
    print(f"Total moves:          {total_moves}")
    print(f"Average time per move: {avg_move_time:.4f} seconds")
    print(f"Median time per move:  {median_move_time:.4f} seconds")
    print(f"Fastest move:         {min_move_time:.4f} seconds")
    print(f"Slowest move:         {max_move_time:.4f} seconds")
    print(f"Moves per second:     {total_moves / total_time:.2f}")

    if len(move_times) > 20:
        early_moves = statistics.mean(move_times[:10])
        late_moves = statistics.mean(move_times[-10:])
        print(f"\nFirst 10 moves avg:   {early_moves:.4f} seconds")
        print(f"Last 10 moves avg:    {late_moves:.4f} seconds")


if __name__ == "__main__":
    main()
