import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from ..go import Board, GameState, Move, Player, Point
from ..go.scoring import DEFAULT_KOMI

PASS = -1


class InvalidGame(ValueError):
    pass


@dataclass
class GameRecord:
    board_size: int
    komi: float
    black_setup: List[Tuple[int, int]] = field(default_factory=list)
    white_setup: List[Tuple[int, int]] = field(default_factory=list)
    first_player: Player = Player.black
    moves: List[Optional[Tuple[int, int]]] = field(default_factory=list)
    winner: Optional[Player] = None

    def key(self) -> str:
        """identity used for de-duplication and the train/val/test split"""
        text = json.dumps(
            [self.board_size, self.black_setup, self.white_setup, self.moves]
        )
        return hashlib.sha1(text.encode()).hexdigest()


def _sgf_point(value: str, board_size: int) -> Optional[Tuple[int, int]]:
    if value == "" or (value == "tt" and board_size <= 19):
        return None
    if len(value) != 2:
        raise InvalidGame(f"bad coordinate {value!r}")
    col = ord(value[0]) - ord("a") + 1
    row = ord(value[1]) - ord("a") + 1
    if not (1 <= row <= board_size and 1 <= col <= board_size):
        raise InvalidGame(f"coordinate {value!r} outside the board")
    return row, col


def _parse_winner(result: str) -> Optional[Player]:
    result = result.strip().upper()
    if result.startswith("B+"):
        return Player.black
    if result.startswith("W+"):
        return Player.white
    return None


def _sgf_nodes(sgf: str) -> List[dict]:
    """
    split an SGF game into nodes of {property: [values]}
    values are read as a whole, so text inside comments or player names
    can never be mistaken for a move; variations are not supported
    """
    nodes = []
    props = None
    ident = ""
    reading_ident = False
    i, n = 0, len(sgf)
    while i < n:
        ch = sgf[i]
        if ch == "[":
            j = i + 1
            value = []
            while j < n and sgf[j] != "]":
                if sgf[j] == "\\" and j + 1 < n:
                    j += 1
                value.append(sgf[j])
                j += 1
            if props is not None and ident:
                props.setdefault(ident, []).append("".join(value))
            reading_ident = False
            i = j + 1
            continue
        if ch == ";":
            props = {}
            nodes.append(props)
            ident = ""
            reading_ident = False
        elif ch.isupper():
            ident = ident + ch if reading_ident else ch
            reading_ident = True
        i += 1
    return nodes


def parse_sgf(sgf: str) -> GameRecord:
    nodes = _sgf_nodes(sgf)
    if not nodes:
        raise InvalidGame("no SGF nodes")

    root = nodes[0]
    board_size = int(root.get("SZ", ["19"])[0].split(":")[0])
    komi = float(root["KM"][0]) if root.get("KM") and root["KM"][0] else DEFAULT_KOMI

    record = GameRecord(board_size=board_size, komi=komi)
    record.winner = _parse_winner(root.get("RE", [""])[0])

    for props in nodes:
        for value in props.get("AB", []):
            point = _sgf_point(value, board_size)
            if point is not None:
                record.black_setup.append(point)
        for value in props.get("AW", []):
            point = _sgf_point(value, board_size)
            if point is not None:
                record.white_setup.append(point)

    if root.get("PL"):
        record.first_player = Player.white if root["PL"][0].upper() == "W" else Player.black
    elif record.black_setup and not record.white_setup:
        record.first_player = Player.white

    expected = record.first_player
    for props in nodes:
        for ident, color in (("B", Player.black), ("W", Player.white)):
            for value in props.get(ident, []):
                if not record.moves:
                    expected = color
                    record.first_player = color
                if color != expected:
                    raise InvalidGame("two moves in a row by the same color")
                record.moves.append(_sgf_point(value, board_size))
                expected = expected.other
    return record


def initial_state(record: GameRecord) -> GameState:
    board = Board(record.board_size)
    for row, col in record.black_setup:
        board.place_stone(Player.black, Point(row, col))
    for row, col in record.white_setup:
        board.place_stone(Player.white, Point(row, col))
    return GameState(board, record.first_player, None, None, komi=record.komi)


def replay(record: GameRecord) -> Iterator[Tuple[GameState, Move]]:
    """yield (state before the move, move) for every move of the game"""
    state = initial_state(record)
    for number, point in enumerate(record.moves, start=1):
        move = Move.pass_turn() if point is None else Move.play(Point(*point))
        if not state.is_valid_move(move):
            raise InvalidGame(f"illegal move {point} at move {number}")
        yield state, move
        state = state.apply_move(move)


def _encode_point(point, board_size):
    return PASS if point is None else (point[0] - 1) * board_size + point[1] - 1


def _decode_point(index, board_size):
    return None if index == PASS else (index // board_size + 1, index % board_size + 1)


def save_games(path: Path, records: List[GameRecord]) -> None:
    """
    flat int16 arrays with per-game offsets; a few bytes per move
    """
    if not records:
        raise ValueError("no games to save")
    board_size = records[0].board_size
    if any(r.board_size != board_size for r in records):
        raise ValueError("all games in one file must share a board size")

    def flatten(lists):
        offsets = np.zeros(len(lists) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(x) for x in lists])
        flat = np.array(
            [_encode_point(p, board_size) for x in lists for p in x], dtype=np.int16
        )
        return flat, offsets

    moves, move_offsets = flatten([r.moves for r in records])
    black, black_offsets = flatten([r.black_setup for r in records])
    white, white_offsets = flatten([r.white_setup for r in records])
    winner = {Player.black: 1, Player.white: -1, None: 0}
    np.savez_compressed(
        path,
        board_size=np.array(board_size),
        moves=moves,
        move_offsets=move_offsets,
        black_setup=black,
        black_offsets=black_offsets,
        white_setup=white,
        white_offsets=white_offsets,
        first_player=np.array([r.first_player.value for r in records], dtype=np.int8),
        komi=np.array([r.komi for r in records], dtype=np.float64),
        winner=np.array([winner[r.winner] for r in records], dtype=np.int8),
    )


class GameArchive:
    """read-only view over a file written by save_games"""

    def __init__(self, path: Path):
        data = np.load(path)
        self.board_size = int(data["board_size"])
        self._arrays = {k: data[k] for k in data.files}

    def __len__(self) -> int:
        return len(self._arrays["komi"])

    def num_moves(self) -> int:
        return int(self._arrays["move_offsets"][-1])

    def _slice(self, name, offsets, index) -> list:
        start, end = self._arrays[offsets][index], self._arrays[offsets][index + 1]
        return [
            _decode_point(int(v), self.board_size)
            for v in self._arrays[name][start:end]
        ]

    def __getitem__(self, index: int) -> GameRecord:
        a = self._arrays
        winner = {1: Player.black, -1: Player.white, 0: None}[int(a["winner"][index])]
        return GameRecord(
            board_size=self.board_size,
            komi=float(a["komi"][index]),
            black_setup=self._slice("black_setup", "black_offsets", index),
            white_setup=self._slice("white_setup", "white_offsets", index),
            first_player=Player(int(a["first_player"][index])),
            moves=self._slice("moves", "move_offsets", index),
            winner=winner,
        )
