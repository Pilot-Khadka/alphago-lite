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
        self.all_games = None
        self.train_games = None
        self.val_games = None

    def _load_all_games(self):
        if self.all_games is None:
            self.all_games = self.read_games_from_file(self.data_dir)
        return self.all_games

    def _create_train_val_split(
        self, train_samples=800, val_samples=200, random_seed=42
    ):
        all_games = self._load_all_games()
        random.seed(random_seed)
        shuffled_games = all_games.copy()
        random.shuffle(shuffled_games)

        total_needed = train_samples + val_samples
        if len(shuffled_games) < total_needed:
            ratio = len(shuffled_games) / total_needed
            train_samples = int(train_samples * ratio)
            val_samples = len(shuffled_games) - train_samples

        self.train_games = shuffled_games[:train_samples]
        self.val_games = shuffled_games[train_samples : train_samples + val_samples]

        print(f"Train games: {len(self.train_games)}")
        print(f"Validation games: {len(self.val_games)}")

    def load_go_data(
        self,
        data_type="train",
        num_samples=1000,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    ):
        if data_type == "train":
            if self.train_games is None:
                raise ValueError(
                    "Train/val split not created. Call create_train_val_loaders first."
                )
            games = self.train_games
        elif data_type == "val":
            if self.val_games is None:
                raise ValueError(
                    "Train/val split not created. Call create_train_val_loaders first."
                )
            games = self.val_games
        else:  # load from all games
            games = self._load_all_games()
            if num_samples < len(games):
                if shuffle:
                    games = random.sample(games, num_samples)
                else:
                    games = games[:num_samples]

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
        all_games = []
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(data_dir, file_name)
                print(f"Reading: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            games = content.split("\n")  # each SGF game is on its own line
            games = [game for game in games if game.strip()]
            all_games.extend(games)

        print(f"Total games: {len(all_games)}")
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
            game, handicap_moves = setup_handicap_game(board_size, handicap_info)

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
        random_seed=42,
    ):
        self._create_train_val_split(train_samples, val_samples, random_seed)
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
