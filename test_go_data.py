import os
import re
from collections import Counter


def extract_board_size(sgf_string):
    match = re.search(r"SZ\[(\d+)\]", sgf_string)
    if match:
        return int(match.group(1))
    return None


def main():
    data_dir = "external/computer-go-dataset/Professional/"
    game_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    board_sizes = Counter()
    total_games = 0

    for file_name in game_files:
        file_path = os.path.join(data_dir, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        games = [g.strip() for g in content.split("\n") if g.strip()]

        for game in games:
            size = extract_board_size(game)
            if size is not None:
                board_sizes[size] += 1
                total_games += 1

    print("Total games:", total_games)
    print("Board size tally:")
    for size, count in board_sizes.items():
        print(f"{size}x{size}: {count}")


if __name__ == "__main__":
    main()
