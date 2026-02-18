from typing import Optional
from abc import abstractmethod

from ..go.goboard import Move


class Agent:
    def __init__(self):
        pass

    @abstractmethod
    def select_move(self, game_state) -> Optional[Move]:
        pass
