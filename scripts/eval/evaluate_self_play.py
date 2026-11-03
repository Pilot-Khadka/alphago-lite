from typing import Dict

import json
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from alphago.go import gotypes, goboard
from alphago.agent import Agent, PolicyAgent, RandomAgent
from alphago.runs import best_checkpoint, load_policy


def build_policy_agent(checkpoint_path: Path, device: torch.device) -> tuple[PolicyAgent, int, str]:
    model, encoder, checkpoint = load_policy(checkpoint_path, device)
    label = f"{checkpoint['run_id']}@{checkpoint['step']}"
    return PolicyAgent(model=model, encoder=encoder, device=device), encoder.shape()[1], label


def play_game(
    agent_black: Agent,
    agent_white: Agent,
    board_size: int,
    verbose: bool = False,
) -> Dict:
    bots = {
        gotypes.Player.black: agent_black,
        gotypes.Player.white: agent_white,
    }

    game = goboard.GameState.new_game(board_size)
    total_moves = 0

    while not game.is_over():
        action = bots[game.next_player].select_move(game)
        if verbose and total_moves % 50 == 0:
            print(f"Move {total_moves}: {game.next_player} plays {action}")
        game = game.apply_move(action)
        total_moves += 1

    winner = game.winner()
    return {
        "winner": winner,
        "total_moves": total_moves,
        "black_wins": winner == gotypes.Player.black,
        "white_wins": winner == gotypes.Player.white,
        "result": str(game.result()),
    }


def play_match(agent: Agent, opponent: Agent, board_size: int, num_games: int, label: str) -> Dict:
    as_black = [play_game(agent, opponent, board_size) for _ in tqdm(range(num_games // 2), desc=f"{label} as Black")]
    as_white = [play_game(opponent, agent, board_size) for _ in tqdm(range(num_games - num_games // 2), desc=f"{label} as White")]

    black_wins = sum(r["black_wins"] for r in as_black)
    white_wins = sum(r["white_wins"] for r in as_white)
    wins = black_wins + white_wins

    return {
        "total_games": num_games,
        "wins": wins,
        "losses": num_games - wins,
        "win_rate": wins / num_games,
        "black_win_rate": black_wins / max(len(as_black), 1),
        "white_win_rate": white_wins / max(len(as_white), 1),
        "avg_moves": float(np.mean([r["total_moves"] for r in as_black + as_white])),
    }


class ModelEvaluator:
    def __init__(self, device: torch.device):
        self.device = device

    def evaluate_vs_random(self, checkpoint_path: Path, num_games: int = 50) -> Dict:
        agent, board_size, label = build_policy_agent(checkpoint_path, self.device)
        print(f"\nEvaluating {label} vs Random Agent ({num_games} games)...")
        results = play_match(agent, RandomAgent(board_size), board_size, num_games, "Model")
        return {"model": label, "opponent": "random", **results}

    def evaluate_vs_checkpoint(self, new_checkpoint: Path, old_checkpoint: Path, num_games: int = 50) -> Dict:
        new_agent, board_size, new_label = build_policy_agent(new_checkpoint, self.device)
        old_agent, old_size, old_label = build_policy_agent(old_checkpoint, self.device)
        if board_size != old_size:
            raise ValueError(f"board sizes differ: {board_size} vs {old_size}")
        print(f"\nEvaluating {new_label} vs {old_label} ({num_games} games)...")
        results = play_match(new_agent, old_agent, board_size, num_games, "New")
        return {"model": new_label, "opponent": old_label, **results}

    def learning_curve(self, run_dir: Path, baseline: Path, num_games: int = 50) -> list[Dict]:
        checkpoints = sorted(run_dir.glob("step_*.pth"))
        last_step = json.loads((run_dir / "run.json").read_text())["step"]
        if (run_dir / "last.pth").exists() and not (run_dir / f"step_{last_step:08d}.pth").exists():
            checkpoints.append(run_dir / "last.pth")
        return [self.evaluate_vs_checkpoint(path, baseline, num_games) for path in checkpoints]


def print_evaluation_report(results: Dict, title: str):
    print(f"\n{title}: {results['model']} vs {results['opponent']}")
    print(f"Win Rate:        {results['win_rate']:.2%}")
    print(f"Wins:            {results['wins']}")
    print(f"Losses:          {results['losses']}")
    print(f"Total Games:     {results['total_games']}")
    print(f"As Black:        {results['black_win_rate']:.2%}")
    print(f"As White:        {results['white_win_rate']:.2%}")
    print(f"Avg Moves:       {results['avg_moves']:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Go model performance")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to evaluate")
    parser.add_argument(
        "--eval_type",
        choices=["random", "baseline", "learning_curve"],
        default="baseline",
        help="random: vs random agent, baseline: vs --baseline, learning_curve: every snapshot of --run vs --baseline",
    )
    parser.add_argument("--baseline", type=Path, help="Baseline checkpoint (default: best supervised run)")
    parser.add_argument("--run", type=Path, help="Run directory for learning_curve")
    parser.add_argument("--num_games", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    evaluator = ModelEvaluator(device=torch.device(args.device))
    baseline = args.baseline or best_checkpoint()
    if baseline is None and args.eval_type != "random":
        raise ValueError("--baseline required, no supervised run found")

    if args.eval_type == "learning_curve":
        if not args.run:
            raise ValueError("--run required for learning_curve")
        assert baseline is not None
        results = evaluator.learning_curve(args.run, baseline, args.num_games)
        for r in results:
            print_evaluation_report(r, "Learning curve")
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint required")
        if args.eval_type == "random":
            results = evaluator.evaluate_vs_random(args.checkpoint, args.num_games)
            print_evaluation_report(results, "Evaluation vs Random Agent")
        else:
            assert baseline is not None
            results = evaluator.evaluate_vs_checkpoint(args.checkpoint, baseline, args.num_games)
            print_evaluation_report(results, "Evaluation vs Baseline")

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
