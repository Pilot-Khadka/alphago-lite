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


def get_standard_handicap_positions(board_size, handicap_count):
    if board_size == 19:
        star_points = [
            (3, 3),  # 0: top-left
            (3, 9),  # 1: top-center
            (3, 15),  # 2: top-right
            (9, 3),  # 3: center-left
            (9, 9),  # 4: center
            (9, 15),  # 5: center-right
            (15, 3),  # 6: bottom-left
            (15, 9),  # 7: bottom-center
            (15, 15),  # 8: bottom-right
        ]
    else:
        # only supports 19x19 for now
        raise NotImplementedError()

    # based on handicap count, return num. of appropriate stones
    handicap_sequences = {
        2: [2, 6],  # Upper right, lower left
        3: [2, 6, 8],  # Add lower right
        4: [2, 6, 0, 8],  # Add top left
        5: [2, 6, 0, 8, 4],  # + center
        6: [2, 6, 0, 8, 1, 7],  # + top/bottom center
        7: [2, 6, 0, 8, 1, 7, 4],
        8: [2, 6, 0, 8, 1, 3, 5, 7],
        9: [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }

    if handicap_count in handicap_sequences:
        indices = handicap_sequences[handicap_count]
        return [star_points[i] for i in indices]

    return []


def setup_handicap_game(board_size, handicap_info):
    game = goboard.GameState.new_game(board_size)
    handicap_moves = []

    if not handicap_info["is_handicap_game"]:
        return game, handicap_moves

    # if position in SGF is available, use that
    if handicap_info["handicap_stones"]:
        positions = handicap_info["handicap_stones"]
    else:
        positions = get_standard_handicap_positions(
            board_size, handicap_info["handicap_count"]
        )

    # place handicap stones (always black)
    for row, col in positions:
        point = Point(row, col)
        move = Move.play(point)

        if game.is_valid_move(move):
            game = game.apply_move(move)
            handicap_moves.append(("black", (row, col)))
        else:
            print(f"Warning: Invalid handicap stone at {point}")

    return game, handicap_moves


if __name__ == "__main__":
    folder = "go_games"
    games = read_txt_files(folder)

    moves = extract_moves(games[0])
    winner = extract_winner(games[0])

    for move in moves[:10]:
        print(move)

    print("winner:", winner)
