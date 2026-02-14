from typing import Any

import importlib


class Encoder:
    def name(self) -> str:
        raise NotImplementedError()

    def encode(self, game_state):
        raise NotImplementedError()

    def encode_point(self, point):
        raise NotImplementedError()

    def decode_point_index(self, index):
        raise NotImplementedError()

    def num_points(self):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()


def get_encoder_by_name(name, board_size):
    module = importlib.import_module("src.encoders." + name)
    constructor = getattr(module, "create")
    return constructor(board_size)
