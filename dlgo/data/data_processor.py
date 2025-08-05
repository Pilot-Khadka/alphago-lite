import os
import random
from torch.utils.data import DataLoader

from dlgo.encoders.base import get_encoder_by_name
from dlgo.data.generator import GoGameDataset


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
        self, train_samples=None, val_samples=None, random_seed=42
    ):
        all_games = self._load_all_games()
        random.seed(random_seed)
        shuffled_games = all_games.copy()
        random.shuffle(shuffled_games)

        if train_samples is None or val_samples is None:
            train_samples = int(len(shuffled_games) * 0.9)
            val_samples = int(len(shuffled_games) * (1 - 0.9))

        total_needed = train_samples + val_samples
        if len(shuffled_games) < total_needed:
            ratio = len(shuffled_games) / total_needed
            train_samples = int(train_samples * ratio)
            val_samples = len(shuffled_games) - train_samples

        self.train_games = shuffled_games[:train_samples]
        self.val_games = shuffled_games[train_samples : train_samples + val_samples]

        print("Data split complete:")
        print(f"  - Train games: {len(self.train_games):,}")
        print(f"  - Validation games: {len(self.val_games):,}")
        print(
            f"""  - Total used: {len(self.train_games) + len(self.val_games):,}/{
                len(all_games):,}"""
        )
        print(f"  - Random seed: {random_seed}")

    def read_games_from_file(self, data_dir):
        """Read all SGF games from text files"""
        all_games = []
        for file_name in os.listdir(data_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(data_dir, file_name)
                print(f"Reading: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                games = content.split("\n")
                games = [game for game in games if game.strip()]
                all_games.extend(games)

        print(f"Total games loaded: {len(all_games):,}")
        return all_games

    def create_dataloader(
        self,
        data_type="train",
        batch_size=32,
        shuffle=True,
        num_workers=2,
        max_moves_per_game=None,
        pin_memory=True,
    ):
        """Create a PyTorch DataLoader for the specified data split"""

        if data_type == "train":
            if self.train_games is None:
                raise ValueError(
                    "Train/val split not created. Call create_train_val_split first."
                )
            games = self.train_games
        elif data_type == "val":
            if self.val_games is None:
                raise ValueError(
                    "Train/val split not created. Call create_train_val_split first."
                )
            games = self.val_games
        elif data_type == "all":
            games = self._load_all_games()
        else:
            raise ValueError(
                f"Invalid data_type: {data_type}. Use 'train', 'val', or 'all'"
            )

        dataset = GoGameDataset(games, self.encoder, max_moves_per_game)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True if data_type == "train" else False,
            persistent_workers=True if num_workers > 0 else False,
        )

    def create_train_val_loaders(
        self,
        train_samples=None,
        val_samples=None,
        batch_size=32,
        shuffle_train=True,
        num_workers=2,
        random_seed=42,
        max_moves_per_game=None,
    ):
        self._create_train_val_split(train_samples, val_samples, random_seed)

        train_loader = self.create_dataloader(
            data_type="train",
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            max_moves_per_game=max_moves_per_game,
        )

        val_loader = self.create_dataloader(
            data_type="val",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            max_moves_per_game=max_moves_per_game,
        )

        return train_loader, val_loader
