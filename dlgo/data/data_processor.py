import os
import random
import numpy as np
from torch.utils.data import DataLoader

from dlgo.gotypes import Point
from dlgo.goboard import GameState, Move
from dlgo.data.process_files import extract_moves, setup_handicap_game
from dlgo.encoders.base import get_encoder_by_name
from dlgo.data.generator import GoDataset


class GoDataProcessor:
    def __init__(self, encoder="oneplane", data_directory="go_data"):
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = os.path.join(os.getcwd(), data_directory)

    def load_go_data(
        self,
        data_type="train",
        num_samples=1000,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    ):
        games = self.read_games_from_file(self.data_dir)
        if num_samples < len(games):
            games = random.sample(games, num_samples)
        features, labels = self.process_games(games)
        dataset = GoDataset(features, labels)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=False,
        )

    def read_games_from_file(self, data_dir):
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(data_dir, file_name)
                print(f"\nReading: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    print(content[:500])

            games = content.split("\n")  # each SGF game is on its own line
            # remove empty lines
            games = [game for game in games if game.strip()]

            print(f"Total games: {len(games)}")
        return games

    def process_games(self, games):
        all_features = []
        all_labels = []

        for i, sgf_content in enumerate(games):
            game_data = self.parse_sgf_game(sgf_content)
            if game_data:
                all_features.extend(game_data["features"])
                all_labels.extend(game_data["labels"])

            if i % 100 == 0:
                print(f"Processed {i} games...")

        return np.array(all_features), np.array(all_labels)

    def parse_sgf_game(self, sgf_content):
        board_size = 19
        game = GameState.new_game(board_size)

        features = []
        labels = []

        moves, handicap_info = extract_moves(sgf_content)
        if handicap_info["is_handicap_game"]:
            # setup game with handicap
            game, handicap_moves = setup_handicap_game(
                board_size, handicap_info)

        for move in moves:
            color, (row, col) = move
            point = Point(row, col)
            board_tensor = self.encoder.encode(game)
            move_index = self.encoder.encode_point(point)

            label = np.zeros(self.encoder.num_points())
            label[move_index] = 1

            features.append(board_tensor)
            labels.append(label)

            move_obj = Move.play(point)
            game = game.apply_move(move_obj)
        return {
            "features": np.array(features),
            "labels": np.array(labels),
            "num_moves": len(moves),
            "board_size": board_size,
        }

    def create_train_val_loaders(
        self,
        train_samples=800,
        val_samples=200,
        batch_size=32,
        shuffle_train=True,
        num_workers=2,
    ):
        train_loader = self.load_go_data(
            data_type="train",
            num_samples=train_samples,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
        )

        val_loader = self.load_go_data(
            data_type="val",
            num_samples=val_samples,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        return train_loader, val_loader
