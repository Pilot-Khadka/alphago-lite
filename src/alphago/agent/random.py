import random

from ..go.goboard import Move
from ..go.gotypes import Point
from ..agent.base import Agent
from ..agent.helpers import is_point_an_eye


class RandomAgent(Agent):
    def __init__(self, board_size: int):
        super().__init__()
        self.board_size = board_size

    def select_move(self, game_state):
        candidates = []
        for r in range(1, game_state.board.size + 1):
            for c in range(1, game_state.board.size + 1):
                candidate = Point(row=r, col=c)
                if game_state.is_valid_move(
                    Move.play(candidate)
                ) and not is_point_an_eye(
                    game_state.board, candidate, game_state.next_player
                ):
                    candidates.append(candidate)
        if not candidates:
            return Move.pass_turn()
        return Move.play(random.choice(candidates))
