import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..encoders import get_encoder_by_name
from .games import GameArchive, InvalidGame, replay


def transform_planes(planes: np.ndarray, k: int) -> np.ndarray:
    """one of the 8 board symmetries: k % 4 quarter turns, then a transpose if k >= 4"""
    out = np.rot90(planes, k % 4, axes=(-2, -1))
    if k >= 4:
        out = np.swapaxes(out, -2, -1)
    return out


def symmetry_tables(board_size: int) -> np.ndarray:
    """tables[k][i] = flat index that point i moves to under symmetry k"""
    grid = np.arange(board_size * board_size).reshape(board_size, board_size)
    tables = np.zeros((8, board_size * board_size), dtype=np.int64)
    for k in range(8):
        moved = transform_planes(grid, k).reshape(-1)
        tables[k][moved] = np.arange(board_size * board_size)
    return tables


class PositionStream(IterableDataset):
    """
    replays games from a GameArchive and encodes positions on the fly
    each DataLoader worker gets its own subset of games
    """

    def __init__(
        self,
        archive_path: Path,
        encoder_name: str,
        encoder_kwargs: Optional[dict] = None,
        include_pass: bool = False,
        augment: bool = True,
        shuffle: bool = True,
        shuffle_buffer: int = 20000,
        loop: bool = True,
        sample_every: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        self.archive_path = Path(archive_path)
        self.encoder_name = encoder_name
        self.encoder_kwargs = encoder_kwargs or {}
        self.include_pass = include_pass
        self.augment = augment
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer if shuffle else 0
        self.loop = loop
        self.sample_every = sample_every
        self.seed = seed

    def _positions(self, archive, encoder, game_indices, rng, tables):
        num_points = archive.board_size * archive.board_size
        for game_index in game_indices:
            try:
                for number, (state, move) in enumerate(replay(archive[game_index])):
                    if number % self.sample_every:
                        continue
                    if move.is_pass:
                        if not self.include_pass:
                            continue
                        label = num_points
                    else:
                        label = encoder.encode_point(move.point)

                    planes = encoder.encode(state).astype(np.int8)
                    if self.augment:
                        k = rng.randrange(8)
                        planes = transform_planes(planes, k)
                        if label < num_points:
                            label = int(tables[k][label])
                    yield np.ascontiguousarray(planes), label
            except InvalidGame:
                continue

    def __iter__(self):
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + worker_id)

        archive = GameArchive(self.archive_path)
        encoder = get_encoder_by_name(
            self.encoder_name, archive.board_size, **self.encoder_kwargs
        )
        tables = symmetry_tables(archive.board_size)
        game_indices = list(range(worker_id, len(archive), num_workers))

        buffer = []
        while True:
            if self.shuffle:
                rng.shuffle(game_indices)
            for planes, label in self._positions(
                archive, encoder, game_indices, rng, tables
            ):
                if not self.shuffle_buffer:
                    yield torch.from_numpy(planes), label
                    continue
                buffer.append((planes, label))
                if len(buffer) >= self.shuffle_buffer:
                    j = rng.randrange(len(buffer))
                    buffer[j], buffer[-1] = buffer[-1], buffer[j]
                    planes, label = buffer.pop()
                    yield torch.from_numpy(planes), label
            if not self.loop:
                break

        rng.shuffle(buffer)
        for planes, label in buffer:
            yield torch.from_numpy(planes), label
