from typing import Dict

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from alphago.go import gotypes, goboard
from alphago.encoders import OnePlaneEncoder
from alphago.networks import SmallResidualNetwork
from alphago.agent import Agent, PolicyAgent, RandomAgent
from alphago.util import load_model


class DLWrapper(nn.Module):
    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, input_arr):
        input_tensor = torch.tensor(input_arr, dtype=torch.float32, device=self.device)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy()


def build_policy_agent(
    encoder: OnePlaneEncoder, checkpoint_path: Path, device: torch.device
) -> PolicyAgent:
    model = SmallResidualNetwork(
        channel_size=encoder.num_planes,
        board_size=encoder.board_size,
    )
    model = load_model(model, checkpoint_path)
    wrapper = DLWrapper(model, device=device)
    return PolicyAgent(model=wrapper, encoder=encoder)


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
    move_history = []

    while not game.is_over():
        bot = bots[game.next_player]
        action = bot.select_move(game)

        if verbose and total_moves % 50 == 0:
            print(f"Move {total_moves}: {game.next_player} plays {action}")

        move_history.append((game.next_player, action))
        game = game.apply_move(action)
        total_moves += 1

    winner = game.winner()
    return {
        "winner": winner,
        "total_moves": total_moves,
        "black_wins": winner == gotypes.Player.black,
        "white_wins": winner == gotypes.Player.white,
        "move_history": move_history,
    }


class ModelEvaluator:
    def __init__(
        self,
        encoder: OnePlaneEncoder,
        device: torch.device,
        board_size: int = 19,
    ):
        self.encoder = encoder
        self.device = device
        self.board_size = board_size

    def evaluate_vs_random(self, checkpoint_path: Path, num_games: int = 50) -> Dict:
        print(f"\nEvaluating vs Random Agent ({num_games} games)...")

        agent = build_policy_agent(self.encoder, checkpoint_path, self.device)
        random_agent = RandomAgent(self.board_size)

        results_as_black = []
        results_as_white = []

        for i in tqdm(range(num_games // 2), desc="As Black"):
            result = play_game(agent, random_agent, self.board_size)
            results_as_black.append(result)

        for i in tqdm(range(num_games // 2), desc="As White"):
            result = play_game(random_agent, agent, self.board_size)
            results_as_white.append(result)

        black_wins = sum(r["black_wins"] for r in results_as_black)
        white_wins = sum(r["white_wins"] for r in results_as_white)
        total_wins = black_wins + white_wins

        avg_moves_black = np.mean([r["total_moves"] for r in results_as_black])
        avg_moves_white = np.mean([r["total_moves"] for r in results_as_white])

        return {
            "total_games": num_games,
            "wins": total_wins,
            "losses": num_games - total_wins,
            "win_rate": total_wins / num_games,
            "black_win_rate": black_wins / (num_games // 2),
            "white_win_rate": white_wins / (num_games // 2),
            "avg_moves_as_black": avg_moves_black,
            "avg_moves_as_white": avg_moves_white,
        }

    def evaluate_vs_checkpoint(
        self,
        new_checkpoint: Path,
        old_checkpoint: Path,
        num_games: int = 50,
    ) -> Dict:
        print(
            f"\nEvaluating {new_checkpoint.name} vs {old_checkpoint.name} ({num_games} games)..."
        )

        new_agent = build_policy_agent(self.encoder, new_checkpoint, self.device)
        old_agent = build_policy_agent(self.encoder, old_checkpoint, self.device)

        results_as_black = []
        results_as_white = []

        for i in tqdm(range(num_games // 2), desc="New as Black"):
            result = play_game(new_agent, old_agent, self.board_size)
            results_as_black.append(result)

        for i in tqdm(range(num_games // 2), desc="New as White"):
            result = play_game(old_agent, new_agent, self.board_size)
            results_as_white.append(result)

        black_wins = sum(r["black_wins"] for r in results_as_black)
        white_wins = sum(r["white_wins"] for r in results_as_white)
        total_wins = black_wins + white_wins

        return {
            "total_games": num_games,
            "new_wins": total_wins,
            "old_wins": num_games - total_wins,
            "win_rate": total_wins / num_games,
            "black_win_rate": black_wins / (num_games // 2),
            "white_win_rate": white_wins / (num_games // 2),
        }


def print_evaluation_report(results: Dict, title: str):
    print(f"{title}")
    if "win_rate" in results:
        print(f"Win Rate:        {results['win_rate']:.2%}")
        print(f"Wins:            {results.get('wins', results.get('new_wins', 0))}")
        print(f"Losses:          {results.get('losses', results.get('old_wins', 0))}")
        print(f"Total Games:     {results['total_games']}")

        if "black_win_rate" in results:
            print(f"\nAs Black:         {results['black_win_rate']:.2%}")
            print(f"As White:           {results['white_win_rate']:.2%}")

        if "avg_moves_as_black" in results:
            print(f"\nAvg Moves (Black):    {results['avg_moves_as_black']:.1f}")
            print(f"Avg Moves (White):      {results['avg_moves_as_white']:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Go model performance")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint to evaluate",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["random", "baseline", "tournament", "learning_curve"],
        default="random",
        help="Type of evaluation",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Baseline checkpoint for comparison",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory containing iteration checkpoints",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=10,
        help="Number of games to play",
    )
    parser.add_argument(
        "--board_size",
        type=int,
        default=19,
        help="Board size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = OnePlaneEncoder(args.board_size)

    evaluator = ModelEvaluator(
        encoder=encoder,
        device=device,
        board_size=args.board_size,
    )

    results = {}

    if args.eval_type == "random":
        results = evaluator.evaluate_vs_random(Path(args.checkpoint), args.num_games)
        print_evaluation_report(results, "Evaluation vs Random Agent")

    elif args.eval_type == "baseline":
        if not args.baseline:
            raise ValueError("--baseline required for baseline evaluation")

        results = evaluator.evaluate_vs_checkpoint(
            Path(args.checkpoint),
            Path(args.baseline),
            args.num_games,
        )
        print_evaluation_report(results, "Evaluation vs Baseline")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
