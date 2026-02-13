from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, join_room

import uuid
import time
import eventlet
from enum import Enum


# pyrefly: ignore [missing-import]
from alphago.gotypes import Player, Point
# pyrefly: ignore [missing-import]
from alphago.goboard import GameState, Move
# pyrefly: ignore [missing-import]
from alphago.agent.naive import RandomBot
# pyrefly: ignore [missing-import]
from alphago.agent.human_player import HumanPlayer
# pyrefly: ignore [missing-import]
from alphago.configs.types import GameAnalysis, AIModelType

# Use eventlet for async support with SocketIO
eventlet.monkey_patch()


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

games: Dict[str, "GameSession"] = {}


class GameStatus(Enum):
    WAITING = "waiting"
    PLAYING = "playing"
    PAUSED = "paused"
    FINISHED = "finished"


@dataclass
class MoveData:
    player: str
    move_type: str
    position: Optional[Dict[str, int]] = None
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
    last_modified: float = 0.0

    def to_dict(self):
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
                    # pyrefly: ignore [bad-argument-type]
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
    session = games.get(game_id)
    if (
        not session
        or session.status != GameStatus.PLAYING
        or session.game_state.is_over()
    ):
        return

    current_player = session.players.get(session.game_state.next_player)
    if not isinstance(current_player, RandomBot):
        return

    time.sleep(1.5)

    try:
        current_player_before = session.game_state.next_player
        move = current_player.select_move(session.game_state)

        if move:
            session.game_state = session.game_state.apply_move(move)
            session.last_modified = time.time()

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

            socketio.emit("game_update", {"game": session.to_dict()})

            if session.game_state.is_over():
                session.status = GameStatus.FINISHED
            else:
                next_player = session.players.get(session.game_state.next_player)
                if not isinstance(next_player, HumanPlayer):
                    socketio.start_background_task(make_ai_move, game_id)

    except Exception as e:
        print(f"Error making AI move: {e}")


def start_ai_game_loop(game_id: str):
    socketio.start_background_task(make_ai_move, game_id)


@app.route("/api/games", methods=["POST"])
def create_game():
    data = request.get_json()
    board_size = data.get("board_size", 19)
    mode = data.get("mode", "human-human")

    game_id = str(uuid.uuid4())
    game_state = GameState.new_game(board_size=board_size)
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

    if mode == "bot-bot":
        start_ai_game_loop(game_id)

    return jsonify({"game_id": game_id, "game": session.to_dict()})


@app.route("/api/games/<game_id>/moves", methods=["POST"])
def make_move(game_id: str):
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
        else:
            row = data.get("row")
            col = data.get("col")
            if row is None or col is None:
                return jsonify({"error": "Row and col required for play move"}), 400
            point = Point(row=row + 1, col=col + 1)
            move = Move.play(point)

        if not session.game_state.is_valid_move(move):
            return jsonify({"error": "Invalid move"}), 400

        current_player_before = session.game_state.next_player
        session.game_state = session.game_state.apply_move(move)
        session.last_modified = time.time()

        move_data = MoveData(
            player="black" if current_player_before == Player.black else "white",
            move_type=move_type,
            # pyrefly: ignore [unbound-name]
            position={"row": row, "col": col} if move_type == "play" else None,
            timestamp=time.time(),
        )
        session.move_history.append(asdict(move_data))

        socketio.emit("game_update", {"game": session.to_dict()})

        if session.game_state.is_over():
            session.status = GameStatus.FINISHED

        current_player = session.players.get(session.game_state.next_player)
        if (
            not isinstance(current_player, HumanPlayer)
            and not session.game_state.is_over()
        ):
            socketio.start_background_task(make_ai_move, game_id)

        # Instead of returning game state, a success message is enough
        return jsonify({"message": "Move accepted"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/games/<game_id>/pause", methods=["POST"])
def pause_game(game_id: str):
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404

    session = games[game_id]
    data = request.get_json()
    paused = data.get("paused", True)

    if paused:
        session.status = GameStatus.PAUSED
    else:
        session.status = GameStatus.PLAYING
        current_player = session.players.get(session.game_state.next_player)
        if (
            not isinstance(current_player, HumanPlayer)
            and not session.game_state.is_over()
        ):
            socketio.start_background_task(make_ai_move, game_id)

    session.last_modified = time.time()
    socketio.emit("game_update", {"game": session.to_dict()})
    return jsonify({"message": "Game paused/resumed", "game": session.to_dict()})


@app.route("/api/games/<game_id>/analysis", methods=["POST"])
def analyze_game(game_id: str):
    if game_id not in games:
        return jsonify({"error": "Game not found"}), 404

    session = games[game_id]
    session.last_modified = time.time()
    socketio.emit("game_update", {"game": session.to_dict()})
    return jsonify({"message": "Analysis started", "game": session.to_dict()})


@app.route("/api/games", methods=["GET"])
def list_games():
    return jsonify({"games": [session.to_dict() for session in games.values()]})


@app.route("/api/games/<game_id>", methods=["DELETE"])
def delete_game(game_id: str):
    if game_id in games:
        del games[game_id]
        return jsonify({"message": "Game deleted"})
    return jsonify({"error": "Game not found"}), 404


# SocketIO event handlers
@socketio.on("connect")
def on_connect():
    print("Client connected")


@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected")


@socketio.on("join_game")
def on_join_game(data):
    game_id = data.get("room")
    if game_id:
        join_room(game_id)
        print(f"Client joined room: {game_id}")


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
