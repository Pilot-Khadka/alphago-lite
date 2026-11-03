from typing import Tuple

from ..go import Point


class Encoder:
    def name(self) -> str:
        raise NotImplementedError()

    def encode(self, game_state):
        raise NotImplementedError()

    def encode_point(self, point):
        raise NotImplementedError()

    def decode_point_index(self, index) -> Point:
        raise NotImplementedError()

    def num_points(self):
        raise NotImplementedError()

    def shape(self) -> Tuple[int, int, int]:
        # example: (1,19,19)
        raise NotImplementedError()


def get_encoder_by_name(name, board_size, **kwargs):
    from .alphago_encoder import AlphaGoEncoder
    from .oneplane import OnePlaneEncoder
    from .sevenplane import SevenPlaneEncoder

    encoders = {
        "oneplane": OnePlaneEncoder,
        "sevenplane": SevenPlaneEncoder,
        "alphago": AlphaGoEncoder,
    }
    if name not in encoders:
        raise ValueError(f"unknown encoder {name!r}, expected one of {list(encoders)}")
    return encoders[name](board_size, **kwargs)
