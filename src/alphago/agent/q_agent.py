import torch
from torch import nn
import numpy as np

from ..go import goboard
from .helpers import is_point_an_eye
from alphago.agent.base import Agent
from alphago.encoders import Encoder


class QAgent(Agent):
    def __init__(self, model: nn.Module, encoder: Encoder, device: torch.device):
        super().__init__()
        self._model = model
        self._encoder = encoder
        self._device = device
        self.collector = None

    def set_collector(self, collector):
        self.collector = collector

    def set_temperature(self, temperature):
        self.temperature = temperature

    def rank_moves_eps_greedy(self, values):
        if np.random.random() < self.temperature:
            return np.random.permutation(len(values))
        return np.argsort(values)[::-1]

    def select_move(self, game_state):
        encoded_state = self._encoder.encode(game_state)
        state_tensor = (
            torch.tensor(encoded_state, dtype=torch.float32)
            .unsqueeze(0)
            .to(self._device)
        )

        with torch.no_grad():
            logits = self._model(state_tensor)
            logits = logits.squeeze(0).cpu().numpy()

        legal_point_indices = []
        for move in game_state.legal_moves():
            if move.is_play:
                idx = self._encoder.encode_point(move.point)
                legal_point_indices.append(idx)

        if not legal_point_indices:
            return goboard.Move.pass_turn()

        legal_values = logits[legal_point_indices]
        ranked = self.rank_moves_eps_greedy(legal_values)

        for pos in ranked:
            move_idx = legal_point_indices[pos]
            point = self._encoder.decode_point_index(move_idx)

            if game_state.is_valid_move(
                goboard.Move.play(point)
            ) and not is_point_an_eye(game_state.board, point, game_state.next_player):
                if self.collector is not None:
                    self.collector.record_decision(encoded_state, move_idx)

                return goboard.Move.play(point)

        if self.collector is not None:
            self.collector.record_decision(encoded_state, -1)

        return goboard.Move.pass_turn()
