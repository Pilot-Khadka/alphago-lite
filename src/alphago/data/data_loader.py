import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset


class GoDataset(Dataset):
    def __init__(self, data_dir: str, max_samples: int | None = None):
        board_files = sorted(glob.glob(os.path.join(data_dir, "*boards.npy")))
        move_files = sorted(glob.glob(os.path.join(data_dir, "*moves.npy")))

        if not board_files:
            raise FileNotFoundError(f"No *boards.npy files found in {data_dir}")

        if len(board_files) != len(move_files):
            raise RuntimeError("Mismatched number of board and move shard files.")

        self._boards = [np.load(f, mmap_mode="r") for f in board_files]
        self._moves = [np.load(f, mmap_mode="r") for f in move_files]

        lengths = [len(m) for m in self._moves]
        self._offsets = np.cumsum([0] + lengths)
        self._full_total = int(self._offsets[-1])

        if max_samples is not None:
            self._total = min(max_samples, self._full_total)
        else:
            self._total = self._full_total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int):
        if index >= self._total:
            raise IndexError(
                f"Index {index} out of range for dataset of size {self._total}"
            )

        shard = int(np.searchsorted(self._offsets, index, side="right")) - 1
        local_index = index - int(self._offsets[shard])

        board = torch.from_numpy(self._boards[shard][local_index].astype(np.float32))
        move = torch.tensor(int(self._moves[shard][local_index]), dtype=torch.long)
        return board, move
