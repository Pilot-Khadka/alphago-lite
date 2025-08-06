import os
import random
import pickle
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob


class PreprocessedGoDataset(Dataset):
    def __init__(self, data_file):
        self.data_file = data_file

        # opwn HDF5 file, keep it open for fast access
        self.h5_file = h5py.File(data_file, "r")
        self.boards = self.h5_file["boards"]
        self.moves = self.h5_file["moves"]
        self.game_indices = self.h5_file["game_indices"]
        self.move_indices = self.h5_file["move_indices"]

        self.num_samples = self.h5_file.attrs["num_samples"]
        self.board_shape = self.h5_file.attrs["board_shape"]
        self.move_shape = self.h5_file.attrs["move_shape"]

        print(f"Loaded dataset: {self.num_samples:,} samples")
        print(f"Board shape: {self.board_shape}")
        print(f"Move shape: {self.move_shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        board = torch.FloatTensor(self.boards[idx])
        move = torch.FloatTensor(self.moves[idx])
        return board, move

    def __del__(self):
        if hasattr(self, "h5_file"):
            self.h5_file.close()


class ChunkedGoDataset(Dataset):
    def __init__(self, metadata_file):
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)

        self.chunk_files = self.metadata["chunk_files"]
        self.total_samples = self.metadata["total_samples"]
        self.chunk_size = self.metadata["chunk_size"]
        self.num_chunks = self.metadata["num_chunks"]

        self.current_chunk_idx = -1
        self.current_chunk_data = None

        print(
            f"""Loaded chunked dataset: {self.total_samples:,} samples in {
                self.num_chunks
            } chunks"""
        )

    def __len__(self):
        return self.total_samples

    def _load_chunk(self, chunk_idx):
        if self.current_chunk_idx != chunk_idx:
            if self.current_chunk_data is not None:
                self.current_chunk_data.close()

            chunk_file = self.chunk_files[chunk_idx]
            self.current_chunk_data = h5py.File(chunk_file, "r")
            self.current_chunk_idx = chunk_idx

    def __getitem__(self, idx):
        # which chunk this sample belongs to
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        self._load_chunk(chunk_idx)

        # get  sample from current chunk
        board = torch.FloatTensor(self.current_chunk_data["boards"][local_idx])
        move = torch.FloatTensor(self.current_chunk_data["moves"][local_idx])
        return board, move

    def __del__(self):
        if hasattr(self, "current_chunk_data") and self.current_chunk_data is not None:
            self.current_chunk_data.close()


class FastGoDataProcessor:
    def __init__(self, processed_data_directory="processed_go_data"):
        self.processed_dir = os.path.join(os.getcwd(), processed_data_directory)
        self.train_indices = None
        self.val_indices = None
        self.dataset = None

    def load_preprocessed_data(self, data_file=None, metadata_file=None):
        if data_file:
            # singlw file format
            data_path = os.path.join(self.processed_dir, data_file)
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            self.dataset = PreprocessedGoDataset(data_path)

        elif metadata_file:
            # chunked format
            metadata_path = os.path.join(self.processed_dir, metadata_file)
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            self.dataset = ChunkedGoDataset(metadata_path)

        else:
            # auto-detect format
            single_files = glob(os.path.join(self.processed_dir, "*.h5"))
            metadata_files = glob(os.path.join(self.processed_dir, "*metadata.pkl"))

            if len(single_files) == 1 and "chunk" not in single_files[0]:
                print(f"Auto-detected single file format: {single_files[0]}")
                self.dataset = PreprocessedGoDataset(single_files[0])
            elif metadata_files:
                print(f"Auto-detected chunked format: {metadata_files[0]}")
                self.dataset = ChunkedGoDataset(metadata_files[0])
            else:
                raise FileNotFoundError(
                    "No valid preprocessed data found. Expected either a single .h5 file or chunk_metadata.pkl"
                )

    def create_train_val_split(self, train_ratio=0.9, random_seed=42):
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_preprocessed_data first.")

        total_samples = len(self.dataset)
        indices = list(range(total_samples))

        random.seed(random_seed)
        random.shuffle(indices)

        train_size = int(total_samples * train_ratio)

        self.train_indices = indices[:train_size]
        self.val_indices = indices[train_size:]

        print("Data split complete:")
        print(f"  - Train samples: {len(self.train_indices):,}")
        print(f"  - Validation samples: {len(self.val_indices):,}")
        print(f"  - Total samples: {total_samples:,}")
        print(f"  - Random seed: {random_seed}")

    def create_dataloader(
        self,
        data_type="train",
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    ):
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_preprocessed_data first.")

        if data_type == "train":
            if self.train_indices is None:
                raise ValueError(
                    "Train/val split not created. Call create_train_val_split first."
                )
            subset_dataset = torch.utils.data.Subset(self.dataset, self.train_indices)

        elif data_type == "val":
            if self.val_indices is None:
                raise ValueError(
                    "Train/val split not created. Call create_train_val_split first."
                )
            subset_dataset = torch.utils.data.Subset(self.dataset, self.val_indices)

        elif data_type == "all":
            subset_dataset = self.dataset

        else:
            raise ValueError(
                f"Invalid data_type: {data_type}. Use 'train', 'val', or 'all'"
            )

        return DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True if data_type == "train" else False,
            persistent_workers=True if num_workers > 0 else False,
        )

    def create_train_val_loaders(
        self,
        train_ratio=0.9,
        batch_size=32,
        shuffle_train=True,
        num_workers=4,
        random_seed=42,
    ):
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_preprocessed_data first.")

        self.create_train_val_split(train_ratio, random_seed)

        train_loader = self.create_dataloader(
            data_type="train",
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
        )

        val_loader = self.create_dataloader(
            data_type="val",
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        return train_loader, val_loader

    def get_data_info(self):
        if self.dataset is None:
            return "No dataset loaded"

        info = {
            "total_samples": len(self.dataset),
            "board_shape": getattr(self.dataset, "board_shape", "Unknown"),
            "move_shape": getattr(self.dataset, "move_shape", "Unknown"),
            "dataset_type": type(self.dataset).__name__,
        }

        if hasattr(self.dataset, "metadata"):
            info["num_chunks"] = self.dataset.num_chunks
            info["chunk_size"] = self.dataset.chunk_size

        return info


def list_preprocessed_data(processed_data_directory="processed_go_data"):
    processed_dir = os.path.join(os.getcwd(), processed_data_directory)

    if not os.path.exists(processed_dir):
        print(f"Processed data directory not found: {processed_dir}")
        return

    print(f"Preprocessed data in {processed_dir}:")

    single_files = glob(os.path.join(processed_dir, "*.h5"))
    single_files = [f for f in single_files if "chunk" not in f]

    if single_files:
        print("\nSingle file datasets:")
        for file in single_files:
            filename = os.path.basename(file)
            size = os.path.getsize(file) / (1024**3)  # GB
            print(f"  - {filename} ({size:.2f} GB)")

    metadata_files = glob(os.path.join(processed_dir, "*metadata.pkl"))

    if metadata_files:
        print("\nChunked datasets:")
        for metadata_file in metadata_files:
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            total_size = sum(os.path.getsize(f) for f in metadata["chunk_files"]) / (
                1024**3
            )  # GB
            print(
                f"""  - {metadata["num_chunks"]} chunks, {
                    metadata["total_samples"]:,} samples ({total_size:.2f} GB)"""
            )


if __name__ == "__main__":
    list_preprocessed_data()
    processor = FastGoDataProcessor()
    processor.load_preprocessed_data()
    train_loader, val_loader = processor.create_train_val_loaders(
        batch_size=64, num_workers=4
    )

    print("\nData info:", processor.get_data_info())

    print("\nTesting data loading...")
    for batch_idx, (boards, moves) in enumerate(train_loader):
        print(
            f"""Batch {batch_idx}: Board shape: {boards.shape}, Move shape: {
                moves.shape
            }"""
        )
        if batch_idx >= 2:
            break

    print("Fast data loading test complete!")
