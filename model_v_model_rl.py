from typing import Dict
from collections import defaultdict


import time
import statistics
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn

from alphago import gotypes, goboard
from alphago.encoders import OnePlaneEncoder
from alphago.networks import SmallResidualNetwork
from alphago.agent.policy_agent import PolicyAgent


def normalize_ddp_keys(state_dict: dict, expect_ddp: bool = False) -> dict:
    new_sd = {}

    for k, v in state_dict.items():
        original = k
        while k.startswith("module."):
            k = k[len("module.") :]

        if expect_ddp and original.startswith("module."):
            k = f"module.{k}"

        new_sd[k] = v

    return new_sd


def load_model(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    state_dict = normalize_ddp_keys(state_dict, expect_ddp=False)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


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


def build_agent(encoder: OnePlaneEncoder, checkpoint_path: Path, device: torch.device):
    model = SmallResidualNetwork(
        channel_size=encoder.num_planes,
        board_size=encoder.board_size,
    )
    model = load_model(model, checkpoint_path)
    wrapper = DLWrapper(model, device=device)
    return PolicyAgent(model=wrapper, encoder=encoder)


def run_multiple_games(
    agent_black,
    agent_white,
    board_size: int = 19,
    num_games: int = 5,
):
    all_results = []

    for i in tqdm(range(1, num_games + 1), desc="Running Games"):
        result = run_game(agent_black, agent_white, board_size)
        all_results.append(result)

    winners = [res["winner"] for res in all_results]
    total_moves_list = [res["total_moves"] for res in all_results]
    total_times = [res["total_time"] for res in all_results]

    print("\n=== Games Summary ===")
    print(f"Games played: {num_games}")
    print(f"Black wins: {winners.count(gotypes.Player.black)}")
    print(f"White wins: {winners.count(gotypes.Player.white)}")
    print(f"Avg moves/game: {statistics.mean(total_moves_list):.2f}")
    print(f"Avg time/game:  {statistics.mean(total_times):.2f}s")

    rl_summary = defaultdict(lambda: {"total_reward": 0, "num_moves": 0})
    for result in all_results:
        for player, stats in result["rl_stats"].items():
            rl_summary[player]["total_reward"] += stats["total_reward"]
            rl_summary[player]["num_moves"] += stats["num_moves"]

    for player, stats in rl_summary.items():
        avg_reward = (
            stats["total_reward"] / stats["num_moves"] if stats["num_moves"] > 0 else 0
        )
        print(f"Player {player}:")
        print(f"  Total moves:       {stats['num_moves']}")
        print(f"  Total reward:      {stats['total_reward']:.2f}")
        print(f"  Avg reward/move:   {avg_reward:.4f}")


def run_game(agent_black, agent_white, board_size: int = 19):
    bots: Dict[gotypes.Player, PolicyAgent] = {
        gotypes.Player.black: agent_black,
        gotypes.Player.white: agent_white,
    }

    for bot in bots.values():
        if bot.collector:
            bot.collector.begin_episode()

    game = goboard.GameState.new_game(board_size)

    move_times = []
    total_moves = 0
    start_time = time.time()

    while not game.is_over():
        move_start = time.time()

        bot = bots[game.next_player]
        action = bot.select_move(game)

        game = game.apply_move(action)

        move_duration = time.time() - move_start
        move_times.append(move_duration)
        total_moves += 1

    winner = game.winner()
    rl_stats = {}
    for player, bot in bots.items():
        if bot.collector:
            reward = 1 if winner == player else -1
            bot.collector.complete_episode(reward)

            buffer = bot.collector.to_buffer()
            buffer.save(f"experience_{player}.pt")

            total_reward = sum(buffer.rewards)
            avg_reward = (
                total_reward / len(buffer.rewards) if len(buffer.rewards) > 0 else 0
            )
            rl_stats[player] = {
                "total_reward": total_reward,
                "avg_reward_per_move": avg_reward,
                "num_moves": len(buffer.rewards),
            }
        else:
            rl_stats[player] = {
                "total_reward": 0,
                "avg_reward_per_move": 0,
                "num_moves": 0,
            }

    total_time = time.time() - start_time

    return {
        "winner": winner,
        "total_moves": total_moves,
        "total_time": total_time,
        "move_times": move_times,
        "rl_stats": rl_stats,
    }


def main():
    board_size = 19
    checkpoint_path = Path("checkpoint/best_checkpoint.pth")
    num_games = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = OnePlaneEncoder(board_size)

    agent_black = build_agent(encoder, checkpoint_path, device)
    agent_white = build_agent(encoder, checkpoint_path, device)

    run_multiple_games(agent_black, agent_white, board_size, num_games)


if __name__ == "__main__":
    main()
