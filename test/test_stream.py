import numpy as np
import torch
from torch.utils.data import DataLoader

from alphago.data.games import parse_sgf, save_games
from alphago.data.stream import PositionStream, symmetry_tables, transform_planes

GAMES = [
    "(;SZ[9];B[ee];W[cc];B[gc];W[cg];B[gg];W[];B[ec])",
    "(;SZ[9];B[cc];W[gg];B[cg];W[gc];B[ee])",
    "(;SZ[9];B[aa];W[ii];B[ai];W[ia])",
]


def make_archive(tmp_path):
    path = tmp_path / "games.npz"
    save_games(path, [parse_sgf(g) for g in GAMES])
    return path


def test_symmetries_move_points_consistently():
    size = 9
    tables = symmetry_tables(size)
    for index in [0, 5, 40, 80]:
        plane = np.zeros((1, size, size))
        plane[0, index // size, index % size] = 1
        for k in range(8):
            moved = transform_planes(plane, k)
            assert int(moved.reshape(-1).argmax()) == tables[k][index]


def test_symmetries_are_distinct():
    tables = symmetry_tables(9)
    assert len({tuple(t) for t in tables}) == 8


def test_labels_follow_the_games(tmp_path):
    stream = PositionStream(
        make_archive(tmp_path), "oneplane", augment=False, shuffle=False, loop=False
    )
    labels = [label for _, label in stream]
    expected = []
    for sgf in GAMES:
        for point in parse_sgf(sgf).moves:
            if point is not None:
                expected.append((point[0] - 1) * 9 + point[1] - 1)
    assert labels == expected


def test_pass_label_when_included(tmp_path):
    stream = PositionStream(
        make_archive(tmp_path), "oneplane", include_pass=True,
        augment=False, shuffle=False, loop=False,
    )
    assert 81 in [label for _, label in stream]


def test_augmented_label_matches_augmented_board(tmp_path):
    stream = PositionStream(
        make_archive(tmp_path), "oneplane", augment=True, shuffle=False, loop=False
    )
    for planes, label in stream:
        assert planes.shape == (1, 9, 9)
        assert planes.reshape(-1)[label] == 0


def test_workers_cover_every_position_once(tmp_path):
    stream = PositionStream(
        make_archive(tmp_path), "oneplane", augment=False, shuffle=False, loop=False
    )
    single = sorted(label for _, label in stream)
    loader = DataLoader(stream, batch_size=4, num_workers=2)
    multi = sorted(torch.cat([y for _, y in loader]).tolist())
    assert multi == single


def test_sample_every_and_dtype(tmp_path):
    stream = PositionStream(
        make_archive(tmp_path), "alphago", augment=False, shuffle=False,
        loop=False, sample_every=2,
    )
    items = list(stream)
    assert all(p.dtype == torch.int8 and p.shape == (48, 9, 9) for p, _ in items)
    assert len(items) == 4 + 3 + 2
