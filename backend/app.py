from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import threading
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from dlgo.gotypes import Player, Point
from dlgo import gotypes, goboard
from dlgo.goboard import Board, GameState, Move
from dlgo.agent.naive import RandomBot
from dlgo.agent.human_player import HumanPlayer
from dlgo.configs.types import GameAnalysis, GameMode
from dlgo.configs.types import AIModelType

app = Flask(__name__)
CORS(app)

games: Dict[str, "GameSession"] = {}


class GameStatus(Enum):
    WAITING = "waiting"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"


@dataclass
class MoveData:
    player: str
    move_type: str  # "play", "pass", "resign"
    position: Optional[Dict[str, int]] = None  # {"row": int, "col": int}
    timestamp: float = 0.0


@dataclass
class GameSession:
    id: str
    board_size: int
    game_state: GameState
    players: Dict[Player, Any]
    status: GameStatus
    move_history: list
    current_analysis: Optional[GameAnalysis] = None
    last_modified: float = 0.0  # Track when game was last modified

    def to_dict(self):
        """Convert game session to JSON-serializable dict"""
        board_state = []
        for row in range(1, self.board_size + 1):
            board_row = []
            for col in range(1, self.board_size + 1):
                point = Point(row=row, col=col)
                stone = self.game_state.board.get(point)
                if stone == Player.black:
                    board_row.append("black")
                elif stone == Player.white:
                    board_row.append("white")
                else:
                    board_row.append(None)
            board_state.append(board_row)

        return {
            "id": self.id,
            "board_size": self.board_size,
            "board_state": board_state,
            "current_player": "black"
            if self.game_state.next_player == Player.black
            else "white",
            "status": self.status.value,
            "is_over": self.game_state.is_over(),
            "move_history": self.move_history,
            "analysis": asdict(self.current_analysis)
            if self.current_analysis
            else None,
            "last_modified": self.last_modified,
        }


@app.route("/api/games/<game_id>", methods=["GET"])
def get_game(game_id: str):
    """Get current game state"""
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404

    session = games[game_id]
    return jsonify({"game": session.to_dict()})


@app.route("/api/games/<game_id>/moves", methods=["POST"])
def make_move(game_id: str):
    """Make a move in the game"""
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404

    session = games[game_id]
    if session.status != GameStatus.PLAYING:
        return jsonify({"error": "Game is not active"}), 400

    data = request.get_json()
    move_type = data.get("type", "play")

    try:
        if move_type == "pass":
            move = Move.pass_turn()
        elif move_type == "resign":
            move = Move.resign()
        else:  # play
            row = data.get("row")
            col = data.get("col")
            if row is None or col is None:
                return jsonify({"error": "Row and col required for play move"}), 400

            # Convert from 0-based frontend to 1-based backend
            point = Point(row=row + 1, col=col + 1)
            move = Move.play(point)

        # Validate move
        if not session.game_state.is_valid_move(move):
            return jsonify({"error": "Invalid move"}), 400

        # Apply move
        current_player_before = session.game_state.next_player
        session.game_state = session.game_state.apply_move(move)
        session.last_modified = time.time()

        # Record move
        move_data = MoveData(
            player="black" if current_player_before == Player.black else "white",
            move_type=move_type,
            position={"row": row, "col": col} if move_type == "play" else None,
            timestamp=time.time(),
        )
        session.move_history.append(asdict(move_data))

        # Check if game is over
        if session.game_state.is_over():
            session.status = GameStatus.FINISHED

        # If next player is AI, trigger AI move
        current_player = session.players[session.game_state.next_player]
        if (
            not isinstance(current_player, HumanPlayer)
            and not session.game_state.is_over()
        ):
            # Schedule AI move in background
            threading.Thread(target=make_ai_move, args=(game_id,), daemon=True).start()

        return jsonify({"game": session.to_dict()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/games", methods=["POST"])
def create_game():
    data = request.get_json()

    board_size = data.get("board_size", 19)
    mode = data.get("mode", "human-human")

    game_id = str(uuid.uuid4())
    game_state = GameState.new_game(board_size=board_size)

    # create players based on mode
    black_player, white_player = create_players_from_mode(mode)

    session = GameSession(
        id=game_id,
        board_size=board_size,
        game_state=game_state,
        players={Player.black: black_player, Player.white: white_player},
        status=GameStatus.PLAYING,
        move_history=[],
        last_modified=time.time(),
    )

    games[game_id] = session

    # if AI vs AI, start the game loop
    if mode == "bot-bot":
        start_ai_game_loop(game_id)

    return jsonify({"game_id": game_id, "game": session.to_dict()})


@app.route("/api/games/<game_id>/pause", methods=["POST"])
def pause_game(game_id: str):
    """Pause/resume a game"""
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404

    session = games[game_id]
    data = request.get_json()
    paused = data.get("paused", True)

    if paused:
        session.status = GameStatus.PAUSED
    else:
        session.status = GameStatus.PLAYING
        # Resume AI if needed
        current_player = session.players[session.game_state.next_player]
        if (
            not isinstance(current_player, HumanPlayer)
            and not session.game_state.is_over()
        ):
            threading.Thread(target=make_ai_move, args=(game_id,), daemon=True).start()

    session.last_modified = time.time()
    return jsonify({"game": session.to_dict()})


@app.route("/api/games/<game_id>/analysis", methods=["POST"])
def analyze_game(game_id: str):
    """Trigger game analysis"""
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404

    session = games[game_id]

    # This would trigger actual analysis - for now just acknowledge
    session.last_modified = time.time()
    return jsonify({"message": "Analysis started", "game": session.to_dict()})


def create_players_from_mode(mode: str):
    if mode == "human-human":
        return HumanPlayer(Player.black, "Human Black"), HumanPlayer(
            Player.white, "Human White"
        )
    elif mode == "human-bot":
        return HumanPlayer(Player.black, "Human"), RandomBot(
            Player.white, AIModelType["NAIVE"]
        )
    elif mode == "bot-bot":
        return RandomBot(Player.black, AIModelType["NAIVE"]), RandomBot(
            Player.white, AIModelType["NAIVE"]
        )
    else:
        raise ValueError(f"Unknown game mode: {mode}")


def make_ai_move(game_id: str):
    if game_id not in games:
        return

    session = games[game_id]
    if session.status != GameStatus.PLAYING or session.game_state.is_over():
        return

    current_player = session.players[session.game_state.next_player]
    if isinstance(current_player, HumanPlayer):
        return

    time.sleep(0.1)
    try:
        # Double-check game is still active
        if game_id not in games or games[game_id].status != GameStatus.PLAYING:
            return

        current_player_before = session.game_state.next_player
        move = current_player.select_move(session.game_state)

        if move:
            session.game_state = session.game_state.apply_move(move)
            session.last_modified = time.time()

            # Record move
            move_data = MoveData(
                player="black" if current_player_before == Player.black else "white",
                move_type="pass"
                if move.is_pass
                else "resign"
                if move.is_resign
                else "play",
                position={"row": move.point.row - 1, "col": move.point.col - 1}
                if move.point
                else None,
                timestamp=time.time(),
            )
            session.move_history.append(asdict(move_data))

            # check if game is over
            if session.game_state.is_over():
                session.status = GameStatus.FINISHED
            else:
                # if next player is also AI, schedule another move
                next_player = session.players[session.game_state.next_player]
                if not isinstance(next_player, HumanPlayer):
                    threading.Thread(
                        target=make_ai_move, args=(game_id,), daemon=True
                    ).start()
    except Exception as e:
        print(f"Error making AI move: {e}")


def start_ai_game_loop(game_id: str):
    threading.Thread(target=make_ai_move, args=(game_id,), daemon=True).start()


@app.route("/api/games", methods=["GET"])
def list_games():
    """List all active games"""
    return jsonify({"games": [session.to_dict() for session in games.values()]})


@app.route("/api/games/<game_id>", methods=["DELETE"])
def delete_game(game_id: str):
    """Delete a game"""
    if game_id in games:
        del games[game_id]
        return jsonify({"message": "Game deleted"})
    return jsonify({"error": "Game not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
