import time
import random
import threading

from dlgo.gotypes import Player
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_and_eye
from dlgo.goboard import Move
from dlgo.gotypes import Point
from dlgo.configs.types import AIModelType


class RandomBot(Agent):
    def __init__(
        self, player_color: Player, model_type: AIModelType = None, name: str = ""
    ):
        self.player_color = player_color
        self.model_type = model_type or AIModelType.NAIVE
        self.name = name or f"AI ({self.model_type.value})"

        # AI-specific attributes
        self.model = None
        self.mcts_simulations = 1600
        self.temperature = 1.0
        self.analysis_callback = None

        # Move computation
        self.pending_move = None
        self.computing = False
        self.computation_thread = None

    def start_analysis(self, position, callback):
        """Start continuous analysis of position"""
        self.analysis_callback = callback
        # TODO: Implement continuous analysis

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

    def _start_move_computation(self, game_state):
        """Start computing move in background thread"""
        self.computing = True

        # TODO: Implement actual AI move computation
        # For now, simulate with timer
        def simulate_computation():
            time.sleep(1.0)  # Simulate computation time
            # Generate dummy move (pass for now)
            self.pending_move = goboard.Move.pass_turn()

        self.computation_thread = threading.Thread(target=simulate_computation)
        self.computation_thread.start()
