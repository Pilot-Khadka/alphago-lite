import torch
from tqdm import tqdm
import numpy as np

from dlgo.gotypes import Point
from torch.utils.data import Dataset
from dlgo.goboard import GameState, Move
from dlgo.data.process_files import extract_moves, setup_handicap_game


class GoGameDataset(Dataset):
    def __init__(self, sgf_games, encoder, max_moves_per_game=None):
        self.encoder = encoder
        self.max_moves_per_game = max_moves_per_game

        self.move_indices = []

        print("Indexing games and moves...")
        for game_idx, sgf_content in enumerate(tqdm(sgf_games, desc="Indexing")):
            try:
                moves, handicap_info = extract_moves(sgf_content)
                num_moves = len(moves)

                if self.max_moves_per_game:
                    num_moves = min(num_moves, self.max_moves_per_game)

                for move_idx in range(num_moves):
                    self.move_indices.append((game_idx, move_idx))

            except Exception as e:
                print(f"Failed to index game {game_idx}: {e}")
                continue

        self.sgf_games = sgf_games
        print(f"Total indexed moves: {len(self.move_indices):,}")

    def __len__(self):
        return len(self.move_indices)

    def __getitem__(self, idx):
        game_idx, move_idx = self.move_indices[idx]
        sgf_content = self.sgf_games[game_idx]

        try:
            board_size = 19
            game = GameState.new_game(board_size)

            moves, handicap_info = extract_moves(sgf_content)

            if handicap_info["is_handicap_game"]:
                game, handicap_moves = setup_handicap_game(board_size, handicap_info)

            for i in range(move_idx):
                color, (row, col) = moves[i]
                point = Point(row, col)
                move_obj = Move.play(point)
                game = game.apply_move(move_obj)

            color, (row, col) = moves[move_idx]
            point = Point(row, col)

            board_tensor = self.encoder.encode(game)

            move_index = self.encoder.encode_point(point)
            label = np.zeros(self.encoder.num_points())
            label[move_index] = 1

            return torch.FloatTensor(board_tensor), torch.FloatTensor(label)

        except Exception as e:
            # return dummy sample if processing fails
            print(f"Error processing game {game_idx}, move {move_idx}: {e}")
            dummy_board = torch.zeros((self.encoder.num_planes, 19, 19))
            dummy_label = torch.zeros(self.encoder.num_points())
            return dummy_board, dummy_label
