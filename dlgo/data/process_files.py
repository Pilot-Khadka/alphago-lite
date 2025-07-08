import os


def sgf_to_coords(sgf_coord):
    if sgf_coord == "" or len(sgf_coord) != 2:
        return None  # pass move or invalid
    col = ord(sgf_coord[0]) - ord("a")
    row = ord(sgf_coord[1]) - ord("a")
    return (row, col)


def extract_moves(sgf_string):
    moves = []
    tokens = sgf_string.split(";")

    for token in tokens:
        if token.startswith("B[") or token.startswith("W["):
            color = "b" if token[0] == "B" else "w"
            coord = token[2:4]
            move = sgf_to_coords(coord)
            moves.append((color, move))

    return moves


def read_txt_files(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            print(f"\nReading: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                print(content[:500])

        games = content.split("\n")  # each SGF game is on its own line
        games = [game for game in games if game.strip()]  # remove empty lines

        print(f"Total games: {len(games)}")
    return games


if __name__ == "__main__":
    folder = "go_games"
    games = read_txt_files(folder)

    moves = extract_moves(games[0])

    for move in moves[:10]:
        print(move)
