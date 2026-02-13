import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset


class GoDataset(Dataset):
    def __init__(self, data_dir: str):
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
        self._total = int(self._offsets[-1])

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        shard = int(np.searchsorted(self._offsets, idx, side="right")) - 1
        local_idx = idx - int(self._offsets[shard])

        board = torch.from_numpy(self._boards[shard][local_idx].astype(np.float32))
        move = torch.tensor(int(self._moves[shard][local_idx]), dtype=torch.long)
        return board, move
