import argparse
import random
import time

from alphago.agent.random import RandomAgent
from alphago.go import GameState, Move, Point


def play_random_game(size, rng, max_moves):
    state = GameState.new_game(size)
    states = [state]
    points = [Point(r, c) for r in range(1, size + 1) for c in range(1, size + 1)]
    for _ in range(max_moves):
        rng.shuffle(points)
        move = next(
            (Move.play(p) for p in points if state.is_valid_move(Move.play(p))),
            Move.pass_turn(),
        )
        state = state.apply_move(move)
        states.append(state)
        if state.is_over():
            break
    return states


def time_per_call(fn, items, repeat=1):
    start = time.perf_counter()
    for _ in range(repeat):
        for item in items:
            fn(item)
    return (time.perf_counter() - start) / (len(items) * repeat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=19)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    start = time.perf_counter()
    games = [play_random_game(args.size, rng, args.max_moves) for _ in range(args.games)]
    gen_time = time.perf_counter() - start
    states = [s for game in games for s in game[:-1]]
    num_moves = len(states)

    pairs = [(s, game[i + 1].last_move) for game in games for i, s in enumerate(game[:-1])]
    apply_t = time_per_call(lambda p: p[0].apply_move(p[1]), pairs)

    probes = [(s, Move.play(Point(r, c))) for s in states[::10]
              for r in range(1, args.size + 1) for c in range(1, args.size + 1)]
    valid_t = time_per_call(lambda p: p[0].is_valid_move(p[1]), probes)

    sample = states[:: max(1, num_moves // 30)]
    legal_t = time_per_call(lambda s: s.legal_moves(), sample)

    agent = RandomAgent(args.size)
    agent_t = time_per_call(agent.select_move, sample)

    print(f"board {args.size}x{args.size}, {args.games} random games, {num_moves} moves")
    print(f"  random game generation  {gen_time / num_moves * 1e3:8.3f} ms/move")
    print(f"  apply_move              {apply_t * 1e6:8.1f} us")
    print(f"  is_valid_move           {valid_t * 1e6:8.1f} us")
    print(f"  legal_moves             {legal_t * 1e3:8.3f} ms")
    print(f"  RandomAgent.select_move {agent_t * 1e3:8.3f} ms")


if __name__ == "__main__":
    main()
