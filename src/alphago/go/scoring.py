from dataclasses import dataclass
from typing import Optional

from .gotypes import Player

DEFAULT_KOMI = 7.5


@dataclass(frozen=True)
class GameResult:
    """
    area scoring: stones on the board + empty points surrounded by one color
    all stones on the board are treated as alive
    """

    black_stones: int
    white_stones: int
    black_territory: int
    white_territory: int
    dame: int
    komi: float

    @property
    def black_score(self) -> float:
        return self.black_stones + self.black_territory

    @property
    def white_score(self) -> float:
        return self.white_stones + self.white_territory + self.komi

    @property
    def margin(self) -> float:
        return abs(self.black_score - self.white_score)

    @property
    def winner(self) -> Optional[Player]:
        if self.black_score > self.white_score:
            return Player.black
        if self.white_score > self.black_score:
            return Player.white
        return None

    def __str__(self) -> str:
        if self.winner is None:
            return "Draw"
        return f"{'B' if self.winner == Player.black else 'W'}+{self.margin:g}"


def compute_game_result(board, komi: float = DEFAULT_KOMI) -> GameResult:
    stones = {Player.black: 0, Player.white: 0}
    territory = {Player.black: 0, Player.white: 0}
    dame = 0
    visited = set()

    for point in board._neighbors:
        string = board._grid.get(point)
        if string is not None:
            stones[string.color] += 1
            continue
        if point in visited:
            continue

        # flood fill the empty region and collect the colors bordering it
        region_size = 0
        borders = set()
        stack = [point]
        visited.add(point)
        while stack:
            current = stack.pop()
            region_size += 1
            for neighbor in board._neighbors[current]:
                neighbor_string = board._grid.get(neighbor)
                if neighbor_string is not None:
                    borders.add(neighbor_string.color)
                elif neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if len(borders) == 1:
            territory[borders.pop()] += region_size
        else:
            dame += region_size

    return GameResult(
        black_stones=stones[Player.black],
        white_stones=stones[Player.white],
        black_territory=territory[Player.black],
        white_territory=territory[Player.white],
        dame=dame,
        komi=komi,
    )
