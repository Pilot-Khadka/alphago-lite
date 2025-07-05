import random
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_and_eye
from dlgo.goboard_slow import Move
from dlgo.gotypes import Point


class RandomBot(Agent):
    def select_move(self, game_state):
        """
        Choose random valid move that preserves own eye
        """
        candidates = []
        for r in range(1, game_state.board.size + 1):
            for c in range(1, game_state.board.size + 1):
                candidate = Point(row=r, col=c)
                if game_state.is_valid_move(
                    Move.play(candidate)
                ) and not is_point_and_eye(
                    game_state.board, candidate, game_state.next_player
                ):
                    candidates.append(candidate)
        if not candidates:
            return Move.pass_turn()
        return Move.play(random.choice(candidates))
