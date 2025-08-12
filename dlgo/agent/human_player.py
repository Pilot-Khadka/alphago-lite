from dlgo.agent.base import Agent
from dlgo.gotypes import Player


class HumanPlayer(Agent):
    def __init__(self, player_color: Player, name: str = ""):
        self.player_color = player_color
        self.name = name or f"{player_color.name.title()} Player"
        self.pending_move = None
        self.move_ready = False

    def select_move(self, game_state):
        if self.move_ready and self.pending_move:
            move = self.pending_move
            self.pending_move = None
            self.move_ready = False
            return move
        return None

    def set_move(self, move):
        self.pending_move = move
        self.move_ready = True
