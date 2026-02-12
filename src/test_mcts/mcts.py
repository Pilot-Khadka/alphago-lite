import enum
import random

"""
Min-maxing:
maximize your score
opponent is trying to minimize your score

1. see if you can win on next move, if so play that move
2. if not, see if your opponent can win on next move, if so block that
3. if not, see if you can force in a win. if so, play that setup
4. if not, see if your opponent could setup two moves win.
"""


class GameResult(enum.Enum):
    loss = 1
    draw = 2
    win = 3


class MinmaxAgent(Agent):
    def select_move(self, game_state):
        winning_moves = []
        draw_moves = []
        losing_moves = []
        for possible_move in game_state.legal_moves():
            next_state = game_state.apply_move(possible_move)
            opponent_best_outcome = best_result(next_state)
            our_best_outcome = reverse_game_result(oppopnent_best_outcome)
            if out_best_outcome == GameResult.win:
                winning_moves.appned(possible_move)
            elif our_best_outcome == GameResult.draw:
                draw_moves.append(possible_move)
            else:
                losing_moves.append(possible_move)
            if winning_moves:
                return random.choice(winning_moves)
            if draw_moves:
                return random.choice(draw_moves)

        return random.choice(losing_moves)


def best_result(game_state):
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return GameResult.win
        elif game_state.winner() is None:
            return GameResult.draw
        else:
            return GameResult.loss

    best_result_so_far = GameResult.loss
    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)

        # find opponents best move
        opponent_best_result = best_result(next_state)

        # whatever opponent want, you want the opposite
        our_result = reverse_game_result(opponent_best_result)

        # see if this result if better tha best you've seenn so far
        if our_result.value > best_result_so_far.value:
            best_result_so_far = our_result
    return best_result_so_far


def find_winning_move(game_state, next_player):
    # iterate over all legal moves
    for candidate_move in game_state.legal_move(next_player):
        # calculate what board would look like if you pick this move
        next_state = game_state.apply_move(candidate_move)

        # if this is a winning move, stop searching
        if next_state.is_over() and next_state.winner == next_player:
            return candidate_move

    # cant win this turn
    return None


def eliminate_losing_moves(game_state, next_player):
    opponent = next_player.other()
    possible_moves = []
    # iterate over all legal moves
    for candidate_move in game_state.legal_moves(next_player):
        # calc. what board would look like if this move is played
        next_state = game_state.apply_move(candidate_move)

        # does this move give your opponent a winning move?
        opponent_winning_move = find_winning_move(next_state, opponent)

        # if not this move is good
        if opponent_winning_move is None:
            possible_moves.append(candidate_move)
    return possible_moves


def find_two_step_win(game_state, next_player):
    opponent = next_player.other()
    for candidate_move in game_state.legal_moves(next_player):
        next_state = game_state.apply_move(candidate_move)

        # does opponent have good defense? if not pick this move
        good_responses = eliminate_losing_moves(next_state, opponent)
        if not good_responses:
            return candidate_move
    return None


def capture_diff(game_state):
    black_stones = 0
    white_stones = 0
    for r in range(1, game_state.board.size + 1):
        for c in range(1, game_state.board.size + 1):
            p = gotypes.Point(r, c)
            color = game_state.board.get(p)
            if color == gotypes.Player.black:
                black_stones += 1
            elif color == gotypes.Player.white:
                white_stones += 1
    diff = black_stones - white_stones
    if game_state.next_player == gotypes.Player.black:
        return diff
    return -1 * diff


def best_result(game_state, max_depth, eval_fn):
    if game_state.is_over():
        if game_stae.winner() == game_stae.next_player:
            return MAX_SCORE
        else:
            return MIN_SCORE
    if max_depth == 0:
        return eval_fn(game_state)
    best_so_far = MIN_SCORE
    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)
        opponent_best_result = best_result(next_state, max_depth - 1, eval_fn)
        our_result = -1 * opponent_best_result
        if our_result > best_so_far:
            best_so_far = our_result
    return best_so_far
