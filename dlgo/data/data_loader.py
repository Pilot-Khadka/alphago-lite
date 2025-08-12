import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any


class GoGameNPZDataset(Dataset):
    def __init__(
        self,
        game_files: List[str],
        data_directory: str,
        max_samples_per_game: Optional[int] = None,
        shuffle_moves: bool = True,
    ):
        self.game_files = game_files
        self.data_dir = data_directory
        self.max_samples_per_game = max_samples_per_game
        self.shuffle_moves = shuffle_moves

        self.sample_index = []
        self._build_sample_index()

    def _build_sample_index(self):
        print("Building sample index...")

        for game_file in self.game_files:
            game_path = os.path.join(self.data_dir, game_file)

            try:
                with np.load(game_path) as data:
                    num_moves = int(data["num_moves"])

                if self.max_samples_per_game:
                    num_moves = min(num_moves, self.max_samples_per_game)

                for move_idx in range(num_moves):
                    self.sample_index.append((game_file, move_idx))

            except Exception as e:
                print(f"Warning: Could not read game file {game_file}: {e}")
                continue

        print(f"Total samples in dataset: {len(self.sample_index):,}")

        if self.shuffle_moves:
            random.shuffle(self.sample_index)

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        game_file, move_idx = self.sample_index[idx]
        game_path = os.path.join(self.data_dir, game_file)

        try:
            with np.load(game_path) as data:
                board = torch.from_numpy(data["boards"][move_idx]).float()
                move = torch.from_numpy(data["moves"][move_idx]).float()

                board = torch.from_numpy(data["boards"][move_idx]).float()
                move = torch.from_numpy(data["moves"][move_idx]).float()
            return board, move

        except Exception as e:
            print(f"Error loading sample {idx} from {game_file}: {e}")
            return torch.zeros((1, 19, 19)), torch.zeros(361)


class GoDataProcessor:
    def __init__(self, data_directory: str = "processed_go_data"):
        self.data_dir = os.path.join(os.getcwd(), data_directory)
        self.metadata = None
        self.train_games = None
        self.val_games = None
        self.test_games = None

        self._load_metadata()

    def _load_metadata(self):
        metadata_file = os.path.join(self.data_dir, "metadata.json")

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"""Metadata file not found: {
                    metadata_file
                }. Run preprocessing first."""
            )

        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        print(f"Loaded metadata for {self.metadata['total_games']:,} games")
        print(f"Total samples: {self.metadata['total_samples']:,}")
        print(f"Board shape: {self.metadata['board_shape']}")
        print(f"Move shape: {self.metadata['move_shape']}")

    def create_train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        total_games: Optional[int] = None,
    ):
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, val, and test ratios must sum to 1.0")

        all_game_files = [game["filename"] for game in self.metadata["games"]]

        # apply total_games limit if provided
        if total_games is not None:
            total_games = min(total_games, len(all_game_files))
            all_game_files = all_game_files[:total_games]
            print(f"Using a subset of {total_games:,} games for splitting.")

        random.seed(random_seed)
        shuffled_files = all_game_files.copy()
        random.shuffle(shuffled_files)

        total_games = len(shuffled_files)
        train_size = int(total_games * train_ratio)
        val_size = int(total_games * val_ratio)

        self.train_games = shuffled_files[:train_size]
        self.val_games = shuffled_files[train_size : train_size + val_size]
        self.test_games = shuffled_files[train_size + val_size :]

        print("Data split complete:")
        print(f"  - Train games: {len(self.train_games):,}")
        print(f"  - Validation games: {len(self.val_games):,}")
        print(f"  - Test games: {len(self.test_games):,}")
        print(f"  - Random seed: {random_seed}")

        return {
            "train": len(self.train_games),
            "val": len(self.val_games),
            "test": len(self.test_games),
        }

    def create_dataset(
        self,
        data_type: str = "train",
        max_samples_per_game: Optional[int] = None,
        shuffle_moves: bool = True,
    ) -> GoGameNPZDataset:
        if data_type == "train":
            if self.train_games is None:
                raise ValueError(
                    "Train/val/test split not created. Call create_train_val_test_split first."
                )
            game_files = self.train_games
        elif data_type == "val":
            if self.val_games is None:
                raise ValueError(
                    "Train/val/test split not created. Call create_train_val_test_split first."
                )
            game_files = self.val_games
        elif data_type == "test":
            if self.test_games is None:
                raise ValueError(
                    "Train/val/test split not created. Call create_train_val_test_split first."
                )
            game_files = self.test_games
        elif data_type == "all":
            game_files = [game["filename"] for game in self.metadata["games"]]
        else:
            raise ValueError(
                f"""Invalid data_type: {
                    data_type
                }. Use 'train', 'val', 'test', or 'all'"""
            )

        return GoGameNPZDataset(
            game_files=game_files,
            data_directory=self.data_dir,
            max_samples_per_game=max_samples_per_game,
            shuffle_moves=shuffle_moves,
        )

    def create_dataloader(
        self,
        data_type: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 2,
        max_samples_per_game: Optional[int] = None,
        pin_memory: bool = True,
        drop_last: bool = None,
    ) -> DataLoader:
        if drop_last is None:
            drop_last = True if data_type == "train" else False

        shuffle_moves = shuffle if data_type == "train" else False

        dataset = self.create_dataset(
            data_type=data_type,
            max_samples_per_game=max_samples_per_game,
            shuffle_moves=shuffle_moves,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=True if num_workers > 0 else False,
        )

    def create_train_val_loaders(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        batch_size: int = 32,
        shuffle_train: bool = True,
        num_workers: int = 2,
        random_seed: int = 42,
        max_samples_per_game: Optional[int] = None,
        pin_memory: bool = True,
        total_games: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        if self.train_games is None or self.val_games is None:
            self.create_train_val_test_split(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed,
                total_games=total_games,
            )

        train_loader = self.create_dataloader(
            data_type="train",
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            max_samples_per_game=max_samples_per_game,
            pin_memory=pin_memory,
        )

        val_loader = self.create_dataloader(
            data_type="val",
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=num_workers,
            max_samples_per_game=max_samples_per_game,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader

    def create_train_val_test_loaders(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 32,
        shuffle_train: bool = True,
        num_workers: int = 2,
        random_seed: int = 42,
        max_samples_per_game: Optional[int] = None,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        self.create_train_val_test_split(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        train_loader = self.create_dataloader(
            data_type="train",
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            max_samples_per_game=max_samples_per_game,
            pin_memory=pin_memory,
        )

        val_loader = self.create_dataloader(
            data_type="val",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            max_samples_per_game=max_samples_per_game,
            pin_memory=pin_memory,
        )

        test_loader = self.create_dataloader(
            data_type="test",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            max_samples_per_game=max_samples_per_game,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader

    def get_data_stats(self) -> Dict[str, Any]:
        stats = {
            "total_games": self.metadata["total_games"],
            "total_samples": self.metadata["total_samples"],
            "board_shape": self.metadata["board_shape"],
            "move_shape": self.metadata["move_shape"],
            "encoder_name": self.metadata["encoder_name"],
            "avg_moves_per_game": self.metadata["total_samples"]
            / self.metadata["total_games"],
        }

        if self.train_games is not None:
            stats["split"] = {
                "train_games": len(self.train_games),
                "val_games": len(self.val_games) if self.val_games else 0,
                "test_games": len(self.test_games) if self.test_games else 0,
            }

        return stats

    def load_sample_game(self, game_idx: int = 0) -> Dict[str, np.ndarray]:
        if game_idx >= len(self.metadata["games"]):
            raise IndexError(f"Game index {game_idx} out of range")

        game_info = self.metadata["games"][game_idx]
        game_path = os.path.join(self.data_dir, game_info["filename"])

        with np.load(game_path) as data:
            return {
                "boards": data["boards"],
                "moves": data["moves"],
                "game_idx": int(data["game_idx"]),
                "num_moves": int(data["num_moves"]),
            }

    def verify_data_integrity(self, num_samples: int = 10) -> bool:
        print(f"Verifying data integrity with {num_samples} random samples...")

        game_files = [game["filename"] for game in self.metadata["games"]]
        sample_files = random.sample(game_files, min(num_samples, len(game_files)))

        errors = 0
        for game_file in sample_files:
            try:
                game_path = os.path.join(self.data_dir, game_file)
                with np.load(game_path) as data:
                    boards = data["boards"]
                    moves = data["moves"]

                    if len(boards) != len(moves):
                        print(f"Error: Mismatched array lengths in {game_file}")
                        errors += 1
                        continue

                    if boards.dtype != np.float32 or moves.dtype != np.float32:
                        print(f"Error: Incorrect data types in {game_file}")
                        errors += 1
                        continue

            except Exception as e:
                print(f"Error loading {game_file}: {e}")
                errors += 1

        if errors == 0:
            return True
        else:
            print(f"Found {errors} errors during verification")
            return False


if __name__ == "__main__":
    processor = GoDataProcessor(data_directory="processed_go_data")

    processor.verify_data_integrity()

    train_loader, val_loader = processor.create_train_val_loaders(
        batch_size=64,
        num_workers=4,
        max_samples_per_game=100,  # Limit samples per game
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    print("\nTesting batch loading...")
    for batch_idx, (boards, moves) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Boards shape: {boards.shape}")
        print(f"  Moves shape: {moves.shape}")
        print(f"  Boards dtype: {boards.dtype}")
        print(f"  Moves dtype: {moves.dtype}")

        if batch_idx >= 2:  # Only test a few batches
            break

    stats = processor.get_data_stats()
    print(f"\nData statistics: {stats}")
