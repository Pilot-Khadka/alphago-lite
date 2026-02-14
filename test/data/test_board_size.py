import os

from alphago.data import extract_board_size

DATA_DIR = "external/computer-go-dataset/Professional/"


def test_all_games_are_19x19():
    game_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

    assert len(game_files) > 0, "No dataset files found."

    total_games = 0

    for file_name in game_files:
        file_path = os.path.join(DATA_DIR, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        games = [g.strip() for g in content.split("\n") if g.strip()]

        for game in games:
            size = extract_board_size(game)

            assert size is not None, f"Missing SZ tag in game: {game[:100]}"
            assert size == 19, f"Found non-19 board size: {size}"

            total_games += 1

    assert total_games > 0, "No games were parsed."
