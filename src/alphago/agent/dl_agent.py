import numpy as np

from .. import goboard
from .helpers import is_point_an_eye
from alphago.agent.base import Agent


class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder

    def predict(self, game_state):
        encoded_state = self.encoder.encode(game_state)
        input_tensor = np.array([encoded_state])
        return self.model.predict(input_tensor)

    def select_move(self, game_state):
        num_moves = self.encoder.board_size * self.encoder.board_size
        move_probs = self.predict(game_state)
        move_probs = move_probs**3
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs = move_probs / np.sum(move_probs)
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs
        )

        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(
                goboard.Move.play(point)
            ) and not is_point_an_eye(game_state.board, point, game_state.next_player):
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()
