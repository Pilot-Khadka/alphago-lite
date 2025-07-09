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

    # does this game have handicap?
    handicap_info = extract_handicap_info(sgf_string)

    # Find all B[...] and W[...] patterns
    pattern = r"([BW])\[([a-s]{2}|)\]"
    matches = re.findall(pattern, sgf_string)

    for color_char, coord in matches:
        color = "black" if color_char == "B" else "white"
        move = sgf_to_coords(coord)
        moves.append((color, move))

    return moves, handicap_info


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
    # RE[W+12.5] white wins by 12.5 points, etc
    result_match = re.search(r"RE\[([^\]]+)\]", sgf_string)

    if not result_match:
        return None, None, "No result found"

    result_str = result_match.group(1)

    if result_str == "0" or result_str.upper() == "DRAW":
        return None, 0, "Draw"
    elif result_str == "?" or result_str.upper() == "UNKNOWN":
        return None, None, "Unknown result"
    elif "B+" in result_str.upper():
        # Black wins
        if "R" in result_str.upper():
            return "black", None, "Black wins by resignation"
        else:
            # point margin -> points player won by
            points_match = re.search(r"([0-9.]+)", result_str)
            points = float(points_match.group(1)) if points_match else 0
            return "black", points, f"Black wins by {points} points"
    elif "W+" in result_str.upper():
        # White wins
        if "R" in result_str.upper():
            return "white", None, "White wins by resignation"
        else:
            # point margin -> points player won by
            points_match = re.search(r"([0-9.]+)", result_str)
            points = float(points_match.group(1)) if points_match else 0
            return "white", points, f"White wins by {points} points"

    return None, None, f"Unknown result format: {result_str}"


def extract_handicap_info(sgf_string):
    handicap_info = {
        "handicap_count": 0,
        "handicap_stones": [],
        "komi": 6.5,  # default komi
        "is_handicap_game": False,
    }

    # check for handicap count
    ha_match = re.search(r"HA\[(\d+)\]", sgf_string)
    if ha_match:
        handicap_info["handicap_count"] = int(ha_match.group(1))
        handicap_info["is_handicap_game"] = True

    # check komi
    komi_match = re.search(r"KM\[([0-9.-]+)\]", sgf_string)
    if komi_match:
        handicap_info["komi"] = float(komi_match.group(1))

    # handicap stone position from AB property
    ab_match = re.search(r"AB(\[[a-s]{2}\])+", sgf_string)
    if ab_match:
        # find all coordinate pairs in AB property
        coords = re.findall(r"\[([a-s]{2})\]", ab_match.group(0))
        for coord in coords:
            if len(coord) == 2:
                col = ord(coord[0]) - ord("a") + 1
                row = ord(coord[1]) - ord("a") + 1
                handicap_info["handicap_stones"].append((row, col))

        # handicap stones found, but not HA proprty, count is inferred
        if not handicap_info["is_handicap_game"] and handicap_info["handicap_stones"]:
            handicap_info["handicap_count"] = len(
                handicap_info["handicap_stones"])
            handicap_info["is_handicap_game"] = True

    return handicap_info


if __name__ == "__main__":
    folder = "go_games"
    games = read_txt_files(folder)

    moves = extract_moves(games[0])
    winner = extract_winner(games[0])

    for move in moves[:10]:
        print(move)

    print("winner:", winner)
