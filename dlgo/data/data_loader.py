import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any


class GoGameShardDataset(Dataset):
    def __init__(
        self,
        data_directory: str,
        indices: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
        transform: Optional[callable] = None,
        # return move as index if true; else, convert to one-hot
        move_as_index: bool = True,
        num_move_classes: int = 361,
    ):
        self.data_dir = data_directory
        self.transform = transform
        self.move_as_index = move_as_index
        self.num_move_classes = num_move_classes

        metadata_file = os.path.join(data_directory, "shard_metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Shard metadata not found: {metadata_file}")

        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        self.total_positions = self.metadata["total_positions"]

        if "individual_board_shape" in self.metadata:
            self.board_shape = tuple(self.metadata["individual_board_shape"])
        elif "board_shape" in self.metadata:
            board_shape_raw = self.metadata["board_shape"]
            if isinstance(board_shape_raw, list) and len(board_shape_raw) == 1:
                self.board_shape = tuple(board_shape_raw)
            else:
                self.board_shape = tuple(board_shape_raw)
        else:
            self.board_shape = (361,)

        self.bitpack_boards = self.metadata.get("bitpack_boards", False)

        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(self.total_positions))

        if max_samples is not None and max_samples < len(self.indices):
            self.indices = self.indices[:max_samples]

        self.length = len(self.indices)

        # load global index as memory maps
        index_files = self.metadata["index_files"]
        index_shard_file = os.path.join(data_directory, index_files["shard_mapping"])
        index_offset_file = os.path.join(data_directory, index_files["offset_mapping"])

        self.global_to_shard = np.memmap(index_shard_file, dtype=np.uint16, mode="r")
        self.global_to_offset = np.memmap(index_offset_file, dtype=np.uint32, mode="r")

        self.shard_boards = {}
        self.shard_moves = {}
        self._shard_info = {
            shard["shard_id"]: shard for shard in self.metadata["shards"]
        }

        print("Dataset initialized:")
        print(f"  Individual board shape: {self.board_shape}")
        print(f"  Total positions: {self.total_positions}")
        print(f"  Using indices: {len(self.indices)}")

    def _load_shard(self, shard_id: int):
        """lazy load shard memory maps"""
        if shard_id in self.shard_boards:
            return

        if shard_id not in self._shard_info:
            raise ValueError(
                f"Shard {shard_id} not found in metadata. Available shards: {list(self._shard_info.keys())}"
            )

        shard_info = self._shard_info[shard_id]

        boards_file = os.path.join(self.data_dir, shard_info["boards_file"])
        moves_file = os.path.join(self.data_dir, shard_info["moves_file"])

        if not os.path.exists(boards_file):
            raise FileNotFoundError(f"Boards file not found: {boards_file}")
        if not os.path.exists(moves_file):
            raise FileNotFoundError(f"Moves file not found: {moves_file}")

        try:
            boards_shape = tuple(shard_info["boards_shape"])
            moves_shape = (
                tuple(shard_info["moves_shape"])
                if isinstance(shard_info["moves_shape"], list)
                else (shard_info["moves_shape"],)
            )

            boards_dtype = shard_info["boards_dtype"]
            moves_dtype = shard_info["moves_dtype"]

            if boards_file.endswith(".dat"):
                self.shard_boards[shard_id] = np.memmap(
                    boards_file, dtype=boards_dtype, mode="r", shape=boards_shape
                )
            else:
                self.shard_boards[shard_id] = np.load(boards_file, mmap_mode="r")

            if moves_file.endswith(".dat"):
                self.shard_moves[shard_id] = np.memmap(
                    moves_file, dtype=moves_dtype, mode="r", shape=moves_shape
                )
            else:
                self.shard_moves[shard_id] = np.load(moves_file, mmap_mode="r")

            if self.shard_boards[shard_id].shape != boards_shape:
                raise ValueError(
                    f"Boards shape mismatch: loaded {
                        self.shard_boards[shard_id].shape
                    }, expected {boards_shape}"
                )

            expected_moves_len = boards_shape[0]
            if len(self.shard_moves[shard_id]) != expected_moves_len:
                raise ValueError(
                    f"""Moves length mismatch: loaded {
                        len(self.shard_moves[shard_id])
                    }, expected {expected_moves_len}"""
                )

        except Exception as e:
            print(f"Error loading shard {shard_id}: {e}")
            print(f"  Boards file: {boards_file}")
            print(f"  Moves file: {moves_file}")
            print(f"  Expected boards shape: {shard_info['boards_shape']}")
            print(
                f"""  Expected moves shape: {
                  shard_info.get('moves_shape', 'unknown')}"""
            )
            raise

    def __getitem__(self, idx):
        if idx >= len(self.indices):
            raise IndexError(
                f"""Index {idx} out of range for dataset of size {
                    len(self.indices)}"""
            )

        global_idx = self.indices[idx]
        if global_idx >= len(self.global_to_shard) or global_idx >= len(
            self.global_to_offset
        ):
            raise IndexError(
                f"""Global index {global_idx} out of bounds. Max index: {
                    min(len(self.global_to_shard), len(
                        self.global_to_offset)) - 1
                }"""
            )

        try:
            shard_id = int(self.global_to_shard[global_idx])
            local_offset = int(self.global_to_offset[global_idx])
        except (IndexError, ValueError) as e:
            raise IndexError(
                f"""Error reading global index at {
                             global_idx}: {e}"""
            )

        if shard_id not in self._shard_info:
            available_shards = list(self._shard_info.keys())
            raise ValueError(
                f"""Invalid shard_id {shard_id} for global_idx {
                    global_idx
                }. Available shards: {available_shards}"""
            )

        try:
            self._load_shard(shard_id)
        except Exception as e:
            raise RuntimeError(
                f"""Failed to load shard {
                    shard_id} for global_idx {global_idx}: {e}"""
            )

        shard_size = len(self.shard_boards[shard_id])
        if local_offset >= shard_size:
            raise IndexError(
                f"""Local offset {local_offset} out of bounds for shard {
                    shard_id
                } (size: {shard_size})"""
            )

        try:
            board = self.shard_boards[shard_id][local_offset]
            move_idx = int(self.shard_moves[shard_id][local_offset])
        except (IndexError, ValueError) as e:
            raise RuntimeError(
                f"""Error reading data at shard {
                    shard_id}, offset {local_offset}: {e}"""
            )

        if self.bitpack_boards and isinstance(board, dict):
            board = self._unpack_board_bits(board)
        elif isinstance(board, np.ndarray):
            board = board.astype(np.float32)

        board_tensor = torch.from_numpy(board).float()

        if self.move_as_index:
            move_tensor = torch.tensor(move_idx, dtype=torch.long)
        else:
            move_tensor = torch.zeros(self.num_move_classes, dtype=torch.float32)
            if 0 <= move_idx < self.num_move_classes:
                move_tensor[move_idx] = 1.0

        if self.transform:
            board_tensor = self.transform(board_tensor)

        return board_tensor, move_tensor

    def __len__(self):
        return self.length

    def _unpack_board_bits(self, packed_board_data):
        if isinstance(packed_board_data, dict):
            packed_data = packed_board_data["packed_data"]
            original_shape = packed_board_data["original_shape"]

            unpacked = np.unpackbits(packed_data)
            total_elements = np.prod(original_shape)
            unpacked = unpacked[:total_elements]
            return unpacked.reshape(original_shape).astype(np.float32)
        else:
            return packed_board_data.astype(np.float32)


class GoShardDataProcessor:
    def __init__(self, data_directory: str = "processed_go_data"):
        self.data_dir = os.path.join(os.getcwd(), data_directory)
        self.metadata = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

        self._load_metadata()

    def _load_metadata(self):
        metadata_file = os.path.join(self.data_dir, "shard_metadata.json")

        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f"""Shard metadata file not found: {metadata_file}.Run shard preprocessing first."""
            )

        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        print("Loaded shard metadata:")
        print(f"  Total positions: {self.metadata['total_positions']:,}")
        print(f"  Total shards: {self.metadata['total_shards']}")
        print(f"  Board shape: {self.metadata['individual_board_shape']}")
        print(f"  Board dtype: {self.metadata['board_dtype']}")
        print(f"  Move dtype: {self.metadata['move_dtype']}")
        print(f"  Encoder: {self.metadata['encoder_name']}")

    def create_train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        max_total_samples: Optional[int] = None,
        stratify_by_shard: bool = False,
    ) -> Dict[str, int]:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, val, and test ratios must sum to 1.0")

        total_positions = self.metadata["total_positions"]

        # apply sample limit if specified
        if max_total_samples is not None:
            total_positions = min(max_total_samples, total_positions)
            print(f"Using subset of {total_positions:,} positions for splitting")

        all_indices = list(range(total_positions))

        random.seed(random_seed)
        np.random.seed(random_seed)

        if stratify_by_shard:
            # ensure each shard is represented proportionally
            self._stratified_split(all_indices, train_ratio, val_ratio, test_ratio)
        else:
            random.shuffle(all_indices)

            train_size = int(total_positions * train_ratio)
            val_size = int(total_positions * val_ratio)

            self.train_indices = all_indices[:train_size]
            self.val_indices = all_indices[train_size : train_size + val_size]
            self.test_indices = all_indices[train_size + val_size :]

        print("Data split complete:")
        print(f"  Train samples: {len(self.train_indices):,}")
        print(f"  Val samples: {len(self.val_indices):,}")
        print(f"  Test samples: {len(self.test_indices):,}")
        print(f"  Random seed: {random_seed}")

        return {
            "train": len(self.train_indices),
            "val": len(self.val_indices),
            "test": len(self.test_indices),
        }

    def _stratified_split(self, all_indices, train_ratio, val_ratio, test_ratio):
        shard_indices = {}

        index_files = self.metadata["index_files"]
        index_shard_file = os.path.join(self.data_dir, index_files["shard_mapping"])
        global_to_shard = np.memmap(index_shard_file, dtype=np.uint16, mode="r")

        for idx in all_indices:
            shard_id = int(global_to_shard[idx])
            if shard_id not in shard_indices:
                shard_indices[shard_id] = []
            shard_indices[shard_id].append(idx)

        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        for shard_id, indices in shard_indices.items():
            random.shuffle(indices)

            shard_size = len(indices)
            train_size = int(shard_size * train_ratio)
            val_size = int(shard_size * val_ratio)

            self.train_indices.extend(indices[:train_size])
            self.val_indices.extend(indices[train_size : train_size + val_size])
            self.test_indices.extend(indices[train_size + val_size :])

    def create_dataset(
        self,
        data_type: str = "train",
        max_samples: Optional[int] = None,
        transform: Optional[callable] = None,
        move_as_index: bool = True,
    ) -> GoGameShardDataset:
        if data_type == "train":
            if self.train_indices is None:
                raise ValueError(
                    "Train/val/test split not created. Call create_train_val_test_split first."
                )
            indices = self.train_indices
        elif data_type == "val":
            if self.val_indices is None:
                raise ValueError(
                    "Train/val/test split not created. Call create_train_val_test_split first."
                )
            indices = self.val_indices
        elif data_type == "test":
            if self.test_indices is None:
                raise ValueError(
                    "Train/val/test split not created. Call create_train_val_test_split first."
                )
            indices = self.test_indices
        elif data_type == "all":
            indices = None  # use all data
        else:
            raise ValueError(
                f"Invalid data_type: {data_type}. Use 'train', 'val', 'test', or 'all'"
            )

        return GoGameShardDataset(
            data_directory=self.data_dir,
            indices=indices,
            max_samples=max_samples,
            transform=transform,
            move_as_index=move_as_index,
        )

    def create_dataloader(
        self,
        data_type: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        max_samples: Optional[int] = None,
        pin_memory: bool = True,
        drop_last: Optional[bool] = None,
        transform: Optional[callable] = None,
        move_as_index: bool = True,
        persistent_workers: bool = True,
    ) -> DataLoader:
        if drop_last is None:
            drop_last = True if data_type == "train" else False

        if data_type in ["val", "test"]:
            shuffle = False

        dataset = self.create_dataset(
            data_type=data_type,
            max_samples=max_samples,
            transform=transform,
            move_as_index=move_as_index,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

    def create_train_val_loaders(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        batch_size: int = 32,
        shuffle_train: bool = True,
        num_workers: int = 4,
        random_seed: int = 42,
        max_total_samples: Optional[int] = None,
        pin_memory: bool = True,
        transform: Optional[callable] = None,
        move_as_index: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        if self.train_indices is None or self.val_indices is None:
            self.create_train_val_test_split(
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                random_seed=random_seed,
                max_total_samples=max_total_samples,
            )

        train_loader = self.create_dataloader(
            data_type="train",
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            transform=transform,
            move_as_index=move_as_index,
        )

        val_loader = self.create_dataloader(
            data_type="val",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            transform=transform,
            move_as_index=move_as_index,
        )

        return train_loader, val_loader

    def get_data_stats(self) -> Dict[str, Any]:
        stats = {
            "total_positions": self.metadata["total_positions"],
            "total_shards": self.metadata["total_shards"],
            "board_shape": self.metadata["individual_board_shape"],
            "board_dtype": self.metadata["board_dtype"],
            "move_dtype": self.metadata["move_dtype"],
            "encoder_name": self.metadata["encoder_name"],
            "shard_size": self.metadata["shard_size"],
            "use_float16": self.metadata.get("use_float16", False),
            "bitpack_boards": self.metadata.get("bitpack_boards", False),
        }

        if self.train_indices is not None:
            stats["split"] = {
                "train_samples": len(self.train_indices),
                "val_samples": len(self.val_indices) if self.val_indices else 0,
                "test_samples": len(self.test_indices) if self.test_indices else 0,
            }

        total_size = 0
        for shard in self.metadata["shards"]:
            boards_file = os.path.join(self.data_dir, shard["boards_file"])
            moves_file = os.path.join(self.data_dir, shard["moves_file"])
            if os.path.exists(boards_file):
                total_size += os.path.getsize(boards_file)
            if os.path.exists(moves_file):
                total_size += os.path.getsize(moves_file)

        stats["total_size_gb"] = total_size / (1024**3)
        stats["bytes_per_position"] = (
            total_size / self.metadata["total_positions"]
            if self.metadata["total_positions"] > 0
            else 0
        )

        return stats

    def verify_data_integrity(self, num_samples: int = 1000) -> bool:
        print(f"Verifying shard data integrity with {num_samples} random samples...")

        test_dataset = GoGameShardDataset(
            data_directory=self.data_dir,
            indices=None,
            max_samples=num_samples,
        )

        errors = 0

        try:
            random_indices = random.sample(
                range(len(test_dataset)), min(num_samples, len(test_dataset))
            )

            for i, idx in enumerate(random_indices):
                try:
                    board, move = test_dataset[idx]

                    if not isinstance(board, torch.Tensor):
                        print(f"Error: Board is not a tensor at index {idx}")
                        errors += 1
                        continue

                    if not isinstance(move, torch.Tensor):
                        print(f"Error: Move is not a tensor at index {idx}")
                        errors += 1
                        continue

                    if board.shape != tuple(self.metadata["individual_board_shape"]):
                        print(
                            f"""Error: Incorrect board shape at index {idx}: {
                                board.shape
                            }"""
                        )
                        errors += 1
                        continue

                    if torch.any(torch.isnan(board)) or torch.any(torch.isinf(board)):
                        print(f"Error: NaN/Inf values in board at index {idx}")
                        errors += 1
                        continue

                    if i % 100 == 0:
                        print(f"  Verified {i + 1}/{len(random_indices)} samples...")

                except Exception as e:
                    print(f"Error loading sample {idx}: {e}")
                    errors += 1

        except Exception as e:
            print(f"Error during verification: {e}")
            return False

        if errors == 0:
            print("Data integrity verification passed!")
            return True
        else:
            print(f"Found {errors} errors during verification")
            return False

    def get_sample_batch(
        self, batch_size: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset = self.create_dataset("all", max_samples=batch_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in loader:
            return batch

        raise RuntimeError("Could not load sample batch")


if __name__ == "__main__":
    processor = GoShardDataProcessor(data_directory="processed_go_data")

    if not processor.verify_data_integrity(num_samples=500):
        print("Data integrity check failed!")
        exit(1)

    print("Creating train/val loaders...")
    train_loader, val_loader = processor.create_train_val_loaders(
        train_ratio=0.8,
        val_ratio=0.2,
        batch_size=64,
        num_workers=4,
        max_total_samples=100000,
        move_as_index=True,
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

    print("Testing batch loading...")
    for batch_idx, (boards, moves) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Boards shape: {boards.shape}, dtype: {boards.dtype}")
        print(f"  Moves shape: {moves.shape}, dtype: {moves.dtype}")
        print(f"  Moves range: [{moves.min().item()}, {moves.max().item()}]")
        print(f"  Memory usage: {boards.numel() * 4 / 1024**2:.1f} MB (boards)")

        if batch_idx >= 2:
            break

    stats = processor.get_data_stats()
    print("Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("Testing move as one-hot...")
    onehot_dataset = processor.create_dataset(
        "train", max_samples=100, move_as_index=False
    )
    onehot_loader = DataLoader(onehot_dataset, batch_size=16)

    for boards, moves in onehot_loader:
        print(f"One-hot moves shape: {moves.shape}, dtype: {moves.dtype}")
        print(f"One-hot moves sum: {moves.sum(dim=1)[:5]}")
        break
