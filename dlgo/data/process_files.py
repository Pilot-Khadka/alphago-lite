import re
import os

"""
Each move is ;B[xy] or ;W[xy] where:
    x and y are lowercase letters 'a' to 's' (for 19×19 board).
    a = 0, b = 1, ..., s = 18
"""


def sgf_to_coords(sgf_coord):
    if sgf_coord == "" or len(sgf_coord) != 2:
        return None  # pass move or invalid
    col = ord(sgf_coord[0]) - ord("a") + 1
    row = ord(sgf_coord[1]) - ord("a") + 1
    return (row, col)


def extract_moves(sgf_string):
    moves = []

    # Find all B[...] and W[...] patterns
    pattern = r"([BW])\[([a-s]{2}|)\]"
    matches = re.findall(pattern, sgf_string)

    for color_char, coord in matches:
        color = "black" if color_char == "B" else "white"
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


def extract_winner(sgf_string):
    # look for RE[]
    # RE[B->R] : black wins by resignation
    # RE[W+12.5] white wins by 12.5 points
    result_match = re.search(r"RE\[([^\]]+)\]", sgf_string)

    if not result_match:
        return None, None, "No result found"

    result_str = result_match.group(1)

    # Parse different result formats
    if result_str == "0" or result_str.upper() == "DRAW":
        return None, 0, "Draw"
    elif result_str == "?" or result_str.upper() == "UNKNOWN":
        return None, None, "Unknown result"
    elif "B+" in result_str.upper():
        # Black wins
        if "R" in result_str.upper():
            return "black", None, "Black wins by resignation"
        else:
            # Extract point margin
            points_match = re.search(r"([0-9.]+)", result_str)
            points = float(points_match.group(1)) if points_match else 0
            return "black", points, f"Black wins by {points} points"
    elif "W+" in result_str.upper():
        # White wins
        if "R" in result_str.upper():
            return "white", None, "White wins by resignation"
        else:
            # Extract point margin
            points_match = re.search(r"([0-9.]+)", result_str)
            points = float(points_match.group(1)) if points_match else 0
            return "white", points, f"White wins by {points} points"

    return None, None, f"Unknown result format: {result_str}"


if __name__ == "__main__":
    folder = "go_games"
    games = read_txt_files(folder)

    moves = extract_moves(games[0])
    winner = extract_winner(games[0])

    for move in moves[:10]:
        print(move)

    print("winner:", winner)
