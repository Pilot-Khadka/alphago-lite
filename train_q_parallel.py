from typing import Optional

import argparse
from pathlib import Path
import multiprocessing as mp


import torch
import numpy as np
import torch.nn as nn

from alphago.util import load_model
from alphago.encoders import OnePlaneEncoder, Encoder
from alphago.networks import SmallResidualNetwork
from alphago.agent.policy_agent import ExperienceBuffer
from alphago.train.parallel_self_play import ParallelSelfPlay


class QTrainer:
    def __init__(
        self,
        model: nn.Module,
        encoder: Encoder,
        device: torch.device,
        checkpoint_dir: Path,
        board_size: int = 19,
        temperature: float = 0.05,
        num_workers: Optional[int] = None,
        games_per_worker: int = 8,
    ):
        self.model = model
        self.model.to(device)
        self.encoder = encoder
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.board_size = board_size
        self.temperature = temperature
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.iteration = 0
        self.best_win_rate = 0.0

        self.parallel_self_play = ParallelSelfPlay(
            model=self.model,
            encoder=self.encoder,
            device=self.device,
            board_size=self.board_size,
            temperature=self.temperature,
            num_workers=num_workers,
            games_per_worker=games_per_worker,
        )

    def collect_self_play_experiences(
        self, num_games: int
    ) -> tuple[ExperienceBuffer, dict]:
        print(
            f"\nRunning {num_games} self-play games across {self.parallel_self_play.num_workers} workers "
            f"({self.parallel_self_play.games_per_worker} games/worker in flight)..."
        )
        return self.parallel_self_play.collect(num_games)

    def train_on_experience(
        self,
        experience: ExperienceBuffer,
        learning_rate: float,
        clipnorm: float,
        batch_size: int,
    ) -> float:
        self.model.train()

        actions = torch.tensor(experience.actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(
            experience.rewards, dtype=torch.float32, device=self.device
        )
        states = torch.tensor(
            experience.states, dtype=torch.float32, device=self.device
        )

        num_actions = self.model(states[:1]).shape[1]  # e.g. 361
        valid = (actions >= 0) & (actions < num_actions)
        states = states[valid]
        actions = actions[valid]
        rewards = rewards[valid]

        dataset = torch.utils.data.TensorDataset(states, actions, rewards)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        epoch_loss = 0.0
        for batch_states, batch_actions, batch_rewards in loader:
            optimizer.zero_grad()
            q_values = self.model(batch_states)
            predicted_q = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            loss = criterion(predicted_q, batch_rewards)
            loss.backward()

            if clipnorm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clipnorm)

            optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(loader)

    def save_checkpoint(self, iteration: int, metrics: dict):
        checkpoint_path = self.checkpoint_dir / f"iteration_{iteration}.pth"
        payload = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "best_win_rate": self.best_win_rate,
        }
        torch.save(payload, checkpoint_path)
        torch.save(payload, self.checkpoint_dir / "latest.pth")
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.iteration = checkpoint["iteration"]
        self.best_win_rate = checkpoint.get("best_win_rate", 0.0)
        print(f"Loaded checkpoint from iteration {self.iteration}")
        print(f"Best win rate: {self.best_win_rate:.2%}")

    def train(
        self,
        num_iterations: int,
        games_per_iteration: int,
        learning_rate: float = 0.01,
        clipnorm: float = 5.0,
        batch_size: int = 128,
    ):
        print("Starting Self-Play Training")
        print(f"Iterations:             {num_iterations}")
        print(f"Games per iteration:    {games_per_iteration}")
        print(f"Learning rate:          {learning_rate}")
        print(f"Batch size:             {batch_size}")
        print(f"Temperature:            {self.temperature}")
        print(f"Device:                 {self.device}")
        print(f"Workers:                {self.parallel_self_play.num_workers}")
        print(f"Games/worker in flight: {self.parallel_self_play.games_per_worker}")

        for iteration in range(self.iteration, self.iteration + num_iterations):
            print(f"\nIteration {iteration + 1}/{self.iteration + num_iterations}")

            experience, stats_summary = self.collect_self_play_experiences(
                games_per_iteration
            )

            print(f"Collected {len(experience.states)} total experiences")
            print(f"Average reward: {np.mean(experience.rewards):.4f}")
            print(f"Reward std:     {np.std(experience.rewards):.4f}")

            for player, stats in stats_summary.items():
                print(f"\nPlayer {player}:")
                print(f"  Moves:           {stats['total_moves']}")
                print(f"  Total reward:    {stats['total_reward']:.2f}")
                print(f"  Avg reward/move: {stats['avg_reward']:.4f}")

            print("\nTraining on collected experiences...")
            avg_loss = self.train_on_experience(
                experience, learning_rate, clipnorm, batch_size
            )
            print(f"Training loss: {avg_loss:.6f}")

            metrics = {
                "num_experiences": len(experience.states),
                "avg_reward": float(np.mean(experience.rewards)),
                "std_reward": float(np.std(experience.rewards)),
                "training_loss": avg_loss,
                "stats": {str(k): v for k, v in stats_summary.items()},
            }
            self.save_checkpoint(iteration + 1, metrics)

        print("Self-play training completed!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--games_per_iteration", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--board_size", type=int, default=19)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint/self_play")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--games_per_worker", type=int, default=8)
    return parser.parse_args()


def main():
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = OnePlaneEncoder(args.board_size)

    model = SmallResidualNetwork(
        channel_size=encoder.num_planes,
        board_size=args.board_size,
    )

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        model = load_model(model, checkpoint_path)
        model.to(device)
        print(f"Loaded initial model from {checkpoint_path}")

    trainer = QTrainer(
        model=model,
        encoder=encoder,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir),
        board_size=args.board_size,
        temperature=args.temperature,
        num_workers=args.num_workers,
        games_per_worker=args.games_per_worker,
    )

    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))

    trainer.train(
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
