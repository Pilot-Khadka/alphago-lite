import os
import glob
import random
import shutil


def split_shards(
    output_dir: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    board_files = sorted(glob.glob(os.path.join(output_dir, "*boards.npy")))
    if not board_files:
        raise FileNotFoundError(f"No *boards.npy files found in {output_dir}")

    shard_pairs = []
    for board_file in board_files:
        move_file = board_file.replace("boards.npy", "moves.npy")
        if not os.path.exists(move_file):
            raise FileNotFoundError(f"Missing paired move file for {board_file}")
        shard_pairs.append((board_file, move_file))

    rng = random.Random(seed)
    rng.shuffle(shard_pairs)

    n = len(shard_pairs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        "train": shard_pairs[:n_train],
        "val": shard_pairs[n_train : n_train + n_val],
        "test": shard_pairs[n_train + n_val :],
    }

    for split_name, pairs in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for board_file, move_file in pairs:
            shutil.move(board_file, split_dir)
            shutil.move(move_file, split_dir)
        print(f"{split_name}: {len(pairs)} shards: {split_dir}")
