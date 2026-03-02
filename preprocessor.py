from typing import List


import os
import zipfile
from pathlib import Path


from alphago.data.sharding import split_shards
from alphago.data.preprocessor import GoGamePreprocessor

# from alphago.encoders.oneplane import OnePlaneEncoder
from alphago.encoders.alphago_encoder import AlphaGoEncoder
from alphago.data import extract_board_size


DATA_DIR = Path("external/computer-go-dataset/Professional/")
OUTPUT_DIR = Path("dataset/alphago/")
SHARD_SIZE = 500_000
SPLIT_RATIOS = (0.8, 0.1, 0.1)


def extract_zips(directory: Path) -> None:
    for zip_path in directory.glob("*.zip"):
        print(f"Unzipping: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(directory)


def load_games(directory: Path) -> List[str]:
    all_games = []
    game_files = list(directory.glob("*.txt"))

    for idx, file_path in enumerate(game_files, start=1):
        print(f"Reading file {idx}/{len(game_files)}: {file_path.name}")
        with file_path.open("r", encoding="utf-8") as f:
            lines = (line.strip() for line in f)
            games = [line for line in lines if line]
            all_games.extend(games)

    return all_games


def infer_board_size(games: List[str]) -> int:
    if not games:
        raise ValueError("No games found to infer board size.")

    for game in games:
        size = extract_board_size(game)
        if size:
            return size

    raise ValueError("Could not determine board size from games.")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extract_zips(DATA_DIR)

    all_games = load_games(DATA_DIR)
    board_size = infer_board_size(all_games)

    print(f"Inferred board size: {board_size}")

    # encoder = OnePlaneEncoder(board_size=board_size)
    encoder = AlphaGoEncoder(board_size=board_size)

    preprocessor = GoGamePreprocessor(
        encoder=encoder,
        data_directory=str(DATA_DIR),
        output_directory=str(OUTPUT_DIR),
    )

    num_processes = max(1, (os.cpu_count() or 2) - 2)

    preprocessor.preprocess(
        all_games,
        max_moves_per_game=None,
        num_processes=num_processes,
        shard_size=SHARD_SIZE,
    )

    split_shards(str(OUTPUT_DIR), ratios=SPLIT_RATIOS)


if __name__ == "__main__":
    main()
