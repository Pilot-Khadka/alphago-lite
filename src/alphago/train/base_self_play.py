from typing import Any, Dict, Generic, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn

from alphago.go import gotypes, goboard
from alphago.encoders import Encoder

ExperienceT = TypeVar("ExperienceT")


@dataclass
class GameTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class GameResult:
    winner: gotypes.Player | None
    total_moves: int
    transitions: Dict[gotypes.Player, list[GameTransition]] = field(
        default_factory=dict
    )


class BaseSelfPlayTrainer(ABC, Generic[ExperienceT]):
    def __init__(
        self,
        model: nn.Module,
        encoder: Encoder,
        checkpoint_dir: Path,
        board_size: int = 19,
    ):
        self.model = model
        self.encoder = encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.board_size = board_size
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.model.to(self.device)
        self.iteration = 0

    @abstractmethod
    def build_agent(self) -> Any:
        """Return a move-selection agent wired to the current model."""

    @abstractmethod
    def aggregate_transitions(
        self,
        all_transitions: Dict[gotypes.Player, list[GameTransition]],
    ) -> ExperienceT:
        """
        Convert raw per-player transitions collected across all games into
        whatever structure train_on_experience expects.
        """

    @abstractmethod
    def calculate_loss(
        self,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_targets: torch.Tensor,
    ) -> torch.Tensor:
        """calculate the RL training loss for one mini-batch."""

    @torch.no_grad()
    def predict(self, input_arr: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(input_arr, dtype=torch.float32, device=self.device)
        return self.model(tensor).squeeze(0).cpu().numpy()

    def run_self_play_game(self) -> GameResult:
        agents = {
            gotypes.Player.black: self.build_agent(),
            gotypes.Player.white: self.build_agent(),
        }

        game = goboard.GameState.new_game(self.board_size)
        move_history: list[tuple[gotypes.Player, np.ndarray, int, np.ndarray]] = []

        while not game.is_over():
            player = game.next_player
            state_before = self.encoder.encode(game)
            agent = agents[player]
            action = agent.select_move(game)
            game = game.apply_move(action)
            if not action.is_play:
                continue

            state_after = self.encoder.encode(game)
            encoded_action = self.encoder.encode_point(
                point=action.point if action.point else None
            )
            move_history.append((player, state_before, encoded_action, state_after))

        winner = game.winner()
        num_moves = len(move_history)

        transitions: Dict[gotypes.Player, list[GameTransition]] = {
            gotypes.Player.black: [],
            gotypes.Player.white: [],
        }

        for i, (player, state, action, next_state) in enumerate(move_history):
            is_last = i == num_moves - 1
            if is_last:
                reward = 1.0 if winner == player else -1.0
            else:
                reward = 0.0

            transitions[player].append(
                GameTransition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=is_last,
                )
            )

        return GameResult(
            winner=winner,
            total_moves=num_moves,
            transitions=transitions,
        )

    def collect_self_play_experiences(self, num_games: int) -> tuple[ExperienceT, Dict]:
        all_transitions: Dict[gotypes.Player, list[GameTransition]] = {
            gotypes.Player.black: [],
            gotypes.Player.white: [],
        }
        win_counts: Dict[gotypes.Player, int] = {
            gotypes.Player.black: 0,
            gotypes.Player.white: 0,
        }

        for _ in tqdm(range(num_games), desc="Self-play"):
            result = self.run_self_play_game()
            for player, ts in result.transitions.items():
                all_transitions[player].extend(ts)
            if result.winner is not None:
                win_counts[result.winner] += 1

        experience = self.aggregate_transitions(all_transitions)

        stats = {
            player: {
                "wins": win_counts[player],
                "win_rate": win_counts[player] / num_games,
                "total_moves": len(ts),
            }
            for player, ts in all_transitions.items()
        }

        return experience, stats

    def train_on_experience(
        self,
        experience: ExperienceT,
        learning_rate: float,
        clipnorm: float | None,
        batch_size: int,
    ) -> float:
        self.model.train()

        states, actions, targets = self._experience_to_tensors(experience)

        dataset = torch.utils.data.TensorDataset(states, actions, targets)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        total_loss = 0.0
        for batch_states, batch_actions, batch_targets in loader:
            optimizer.zero_grad()
            loss = self.calculate_loss(batch_states, batch_actions, batch_targets)
            loss.backward()
            if clipnorm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), clipnorm)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @abstractmethod
    def _experience_to_tensors(
        self, experience: ExperienceT
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack the method-specific experience buffer into (states, actions, targets)
        tensors ready for calculate_loss. The targets tensor carries whatever the
        RL method uses discounted returns, TD targets, advantages, etc.
        """

    def save_checkpoint(self, iteration: int, metrics: Dict):
        payload = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
        }
        torch.save(payload, self.checkpoint_dir / f"iteration_{iteration}.pth")
        torch.save(payload, self.checkpoint_dir / "latest.pth")

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.iteration = checkpoint["iteration"]

    def train(
        self,
        num_iterations: int,
        games_per_iteration: int,
        learning_rate: float = 1e-3,
        clipnorm: float | None = 5.0,
        batch_size: int = 128,
    ):
        for iteration in range(self.iteration, self.iteration + num_iterations):
            print(
                f"\n-- Iteration {iteration + 1}/{self.iteration + num_iterations} --"
            )

            experience, stats = self.collect_self_play_experiences(games_per_iteration)
            for player, s in stats.items():
                print(
                    f"  {player}: {s['wins']} wins ({s['win_rate']:.1%}), "
                    f"{s['total_moves']} moves"
                )

            avg_loss = self.train_on_experience(
                experience, learning_rate, clipnorm, batch_size
            )
            print(f"  loss: {avg_loss:.6f}")

            self.save_checkpoint(
                iteration + 1,
                {"loss": avg_loss, "stats": {str(k): v for k, v in stats.items()}},
            )

        print("Training complete.")
