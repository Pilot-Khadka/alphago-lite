import random
import time
import math

# pyrefly: ignore [missing-import]
from alphago.goboard import Move
# pyrefly: ignore [missing-import]
from alphago.gotypes import Point
# pyrefly: ignore [missing-import]
from alphago.agent.base import Agent


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.untried_moves = self._get_valid_moves(self.game_state)

    def _get_valid_moves(self, game_state):
        moves = []

        for row in range(1, game_state.board.size + 1):
            for col in range(1, game_state.board.size + 1):
                candidate = Point(row=row, col=col)
                if not game_state.board.is_empty(candidate):
                    continue
                if game_state.is_valid_move(Move.play(candidate)):
                    moves.append(Move.play(candidate))
        moves.append(Move.pass_turn())
        random.shuffle(moves)
        return moves

    def _is_terminal(self):
        return self.game_state.is_over()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def expand(self):
        if not self.untried_moves:
            return None

        move = self.untried_moves.pop()
        new_game_state = self.game_state.apply_move(move)
        child = MCTSNode(new_game_state, parent=self, move=move)
        self.children[move] = child
        return child

    def simulate(self):
        current_state = self.game_state
        simulation_moves = 0
        max_simulation_moves = 3
        consecutive_passes = 0

        while (
            not current_state.is_over()
            and simulation_moves < max_simulation_moves
            and consecutive_passes < 2
        ):
            valid_moves = self._get_valid_moves(current_state)

            if not valid_moves:
                current_state = current_state.apply_move(Move.pass_turn())
                consecutive_passes += 1
            else:
                move = random.choice(valid_moves)
                current_state = current_state.apply_move(move)
                consecutive_passes = 0
            simulation_moves += 1

        if current_state.is_over():
            winner = current_state.winner()
            if winner == self.game_state.next_player:
                return 1
            elif winner is None:
                return 0.5
            else:
                return 0
        else:
            # Estimate winner based on stone count if simulation incomplete
            return self._estimate_winner(current_state)

    def _estimate_winner(self, game_state):
        black_stones = 0
        white_stones = 0

        for r in range(1, game_state.board.size + 1):
            for c in range(1, game_state.board.size + 1):
                stone = game_state.board.get(Point(row=r, col=c))
                if stone:
                    if stone.name == "black":
                        black_stones += 1
                    else:
                        white_stones += 1

        if game_state.next_player.name == "black":
            return 0.5 + (black_stones - white_stones) * 0.01
        else:
            return 0.5 + (white_stones - black_stones) * 0.01

    def backpropagate(self, result):
        """Propagate simulation result up the tree"""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(1 - result)

    def best_child(self, exploration_weight=1.0):
        """Select best child using UCB1"""
        if not self.children:
            return None

        def ucb1_value(child):
            if child.visits == 0:
                return float("inf")

            exploitation = child.wins / child.visits
            exploration = math.sqrt(math.log(self.visits) / child.visits)
            return exploitation + exploration_weight * exploration

        return max(self.children.values(), key=ucb1_value)


class MCTSBot(Agent):
    def __init__(self, simulation_time=1.0, exploration_weight=1.4):
        self.simulation_time = simulation_time
        self.exploration_weight = exploration_weight

    def select_move(self, game_state):
        if game_state.is_over():
            return Move.pass_turn()

        root = MCTSNode(game_state)
        start_time = time.time()
        simulations = 0

        while time.time() - start_time < self.simulation_time:
            leaf = self._select(root)
            result = leaf.simulate()
            leaf.backpropagate(result)
            simulations += 1

        if not root.children:
            return Move.pass_turn()

        best_child = max(root.children.values(), key=lambda c: c.visits)

        non_pass_children = [c for c in root.children.values() if not c.move.is_pass]
        if non_pass_children:
            best_non_pass = max(non_pass_children, key=lambda c: c.visits)
            if (
                best_child.move.is_pass
                and best_non_pass.visits > best_child.visits * 0.5
            ):
                best_child = best_non_pass

        return best_child.move

    def _select(self, node):
        while not node._is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child(self.exploration_weight)
        return node
