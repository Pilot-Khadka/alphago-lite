import time
import statistics
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from alphago.go import gotypes, goboard
from alphago.encoders import OnePlaneEncoder
from alphago.networks import SmallResidualNetwork
from alphago.agent.dl_agent import DeepLearningAgent
from alphago.util import print_move, GoBoardDisplay
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


def build_agent(encoder: OnePlaneEncoder, checkpoint_path: Path, device: torch.device):
    model = SmallResidualNetwork(
        channel_size=encoder.num_planes,
        board_size=encoder.board_size,
    )
    model = load_model(model, checkpoint_path)
    wrapper = DLWrapper(model, device=device)
    return DeepLearningAgent(model=wrapper, encoder=encoder)


def run_game(agent_black, agent_white, board_size: int = 19):
    bots: Dict[gotypes.Player, DeepLearningAgent] = {
        gotypes.Player.black: agent_black,
        gotypes.Player.white: agent_white,
    }

    game = goboard.GameState.new_game(board_size)
    display = GoBoardDisplay(board_size)

    move_times = []
    total_moves = 0
    start_time = time.time()

    print(f"Starting Go game ({board_size}x{board_size})")
    print("-" * 50)

    while not game.is_over():
        move_start = time.time()

        display.update_board(game.board)
        bot = bots[game.next_player]
        action = bot.select_move(game)

        print_move(game.next_player, action)
        game = game.apply_move(action)

        move_duration = time.time() - move_start
        move_times.append(move_duration)
        total_moves += 1

        if total_moves % 10 == 0:
            avg_last10 = statistics.mean(move_times[-10:])
            print(f"[Move {total_moves:3d}] Avg last 10 moves: {avg_last10:.4f}s")

    # Final stats
    total_time = time.time() - start_time
    print("\nGame Finished.")
    print(f"Total time:      {total_time:.2f}s")
    print(f"Total moves:     {total_moves}")
    print(f"Avg move time:   {statistics.mean(move_times):.4f}s")
    print(f"Median:          {statistics.median(move_times):.4f}s")
    print(f"Fastest:         {min(move_times):.4f}s")
    print(f"Slowest:         {max(move_times):.4f}s")
    print(f"Moves/sec:       {total_moves / total_time:.2f}")

    if len(move_times) >= 20:
        print(f"First 10 avg:    {statistics.mean(move_times[:10]):.4f}s")
        print(f"Last 10 avg:     {statistics.mean(move_times[-10:]):.4f}s")


def main():
    board_size = 19
    checkpoint_path = Path("checkpoint/best_checkpoint.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = OnePlaneEncoder(board_size)

    agent_black = build_agent(encoder, checkpoint_path, device)
    agent_white = build_agent(encoder, checkpoint_path, device)

    run_game(agent_black, agent_white, board_size)


if __name__ == "__main__":
    main()
