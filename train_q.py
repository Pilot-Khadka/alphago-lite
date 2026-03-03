import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from alphago.go import gotypes, goboard
from alphago.encoders import OnePlaneEncoder, Encoder
from alphago.networks import SmallResidualNetwork
from alphago.agent.policy_agent import (
    ExperienceCollector,
    ExperienceBuffer,
)
from alphago.agent.q_agent import QAgent
from alphago.util import load_model


class QTrainer:
    def __init__(
        self,
        model: nn.Module,
        encoder: Encoder,
        device: torch.device,
        checkpoint_dir: Path,
        board_size: int = 19,
        temperature: float = 0.05,
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

    def build_agent(self) -> QAgent:
        agent = QAgent(
            model=self.model,
            encoder=self.encoder,
            device=self.device,
        )
        agent.set_temperature(self.temperature)
        agent.set_collector(ExperienceCollector())
        return agent

    def run_self_play_game(self) -> dict:
        agent_black = self.build_agent()
        agent_white = self.build_agent()

        bots = {
            gotypes.Player.black: agent_black,
            gotypes.Player.white: agent_white,
        }

        for bot in bots.values():
            assert bot.collector is not None
            bot.collector.begin_episode()

        game = goboard.GameState.new_game(self.board_size)
        total_moves = 0

        while not game.is_over():
            bot = bots[game.next_player]
            action = bot.select_move(game)
            game = game.apply_move(action)
            total_moves += 1

        winner = game.winner()

        experiences = {}
        rl_stats = {}

        for player, bot in bots.items():
            reward = 1.0 if winner == player else -1.0
            assert bot.collector is not None
            bot.collector.complete_episode(reward)

            buffer = bot.collector.to_buffer()
            experiences[player] = buffer

            rl_stats[player] = {
                "total_reward": float(np.sum(buffer.rewards)),
                "num_moves": len(buffer.rewards),
                "avg_reward": float(np.mean(buffer.rewards)),
            }

        return {
            "winner": winner,
            "total_moves": total_moves,
            "experiences": experiences,
            "rl_stats": rl_stats,
        }

    def collect_self_play_experiences(
        self, num_games: int
    ) -> tuple[ExperienceBuffer, dict]:
        all_states = []
        all_actions = []
        all_rewards = []
        all_stats = defaultdict(lambda: {"total_reward": 0.0, "num_moves": 0})

        print(f"\nRunning {num_games} self-play games...")
        for _ in tqdm(range(num_games), desc="Self-play"):
            result = self.run_self_play_game()

            for player, buffer in result["experiences"].items():
                all_states.append(buffer.states)
                all_actions.append(buffer.actions)
                all_rewards.append(buffer.rewards)

                stats = result["rl_stats"][player]
                all_stats[player]["total_reward"] += stats["total_reward"]
                all_stats[player]["num_moves"] += stats["num_moves"]

        combined_buffer = ExperienceBuffer(
            states=np.concatenate(all_states),
            actions=np.concatenate(all_actions),
            rewards=np.concatenate(all_rewards),
        )

        stats_summary = {}
        for player, stats in all_stats.items():
            avg_reward = (
                stats["total_reward"] / stats["num_moves"]
                if stats["num_moves"] > 0
                else 0
            )
            stats_summary[player] = {
                "total_moves": stats["num_moves"],
                "total_reward": stats["total_reward"],
                "avg_reward": avg_reward,
            }

        return combined_buffer, stats_summary

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

        # Pass moves recorded as -1 have no valid Q-value target
        valid = actions >= 0
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

            # Raw Q-values, no softmax, since we need unconstrained value estimates
            q_values = self.model(batch_states)

            # Pull out the Q-value for the action actually taken
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

        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
                "best_win_rate": self.best_win_rate,
            },
            checkpoint_path,
        )

        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
                "best_win_rate": self.best_win_rate,
            },
            latest_path,
        )

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

        for iteration in range(self.iteration, self.iteration + num_iterations):
            print(f"Iteration {iteration + 1}/{self.iteration + num_iterations}")

            experience, stats_summary = self.collect_self_play_experiences(
                games_per_iteration
            )

            print(f"\nCollected {len(experience.states)} total experiences")
            print(f"Average reward: {np.mean(experience.rewards):.4f}")
            print(f"Reward std: {np.std(experience.rewards):.4f}")

            for player, stats in stats_summary.items():
                print(f"\nPlayer            {player}:")
                print(f"  Moves:            {stats['total_moves']}")
                print(f"  Total reward:     {stats['total_reward']:.2f}")
                print(f"  Avg reward/move:  {stats['avg_reward']:.4f}")

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
    return parser.parse_args()


def main():
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
