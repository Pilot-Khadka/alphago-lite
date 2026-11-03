import argparse
import asyncio
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from alphago.agent.random import RandomAgent
from alphago.go import GameState, Move, Player, Point
from alphago.runs import best_checkpoint
from alphago.web.bots import load_policy_bot

STATIC_DIR = Path(__file__).parent / "static"
BOARD_SIZES = (9, 13, 19)
BOT_VS_BOT_DELAY = 0.15

app = FastAPI()
bots: dict = {"random": RandomAgent(board_size=19)}


def bot_board_size(bot):
    return None if isinstance(bot, RandomAgent) else bot.board_size


def bot_info(bot):
    return getattr(bot, "info", None)


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.game: GameState | None = None
        self.players = {}
        self.bot_names = {}
        self.paused = False
        self.task = None
        self.send_lock = asyncio.Lock()

    def bot_for(self, player):
        return self.players.get(player)

    def is_bot_turn(self):
        return (
            self.game is not None
            and not self.game.is_over()
            and not self.paused
            and self.bot_for(self.game.next_player) is not None
        )

    def is_human_turn(self):
        return (
            self.game is not None
            and not self.game.is_over()
            and self.bot_for(self.game.next_player) is None
        )

    async def send(self, message):
        async with self.send_lock:
            await self.ws.send_json(message)

    async def error(self, message):
        await self.send({"type": "error", "message": message})

    def stop_bots(self):
        if self.task is not None:
            self.task.cancel()
            self.task = None

    def start_bots(self):
        self.stop_bots()
        if self.is_bot_turn():
            self.task = asyncio.create_task(self.run_bots())

    async def run_bots(self):
        human_present = any(bot is None for bot in self.players.values())
        while self.is_bot_turn():
            await self.send(self.state(thinking=True))
            game = self.game
            assert game is not None
            bot = self.bot_for(game.next_player)
            assert bot is not None
            move = await asyncio.to_thread(bot.select_move, game)
            if not human_present:
                await asyncio.sleep(BOT_VS_BOT_DELAY)
            self.game = game.apply_move(move)
        await self.send(self.state())

    def state(self, thinking=False):
        game = self.game
        assert game is not None
        size = game.board.size
        cells = []
        for row in range(1, size + 1):
            for col in range(1, size + 1):
                color = game.board.get(Point(row, col))
                if color is None:
                    cells.append(".")
                else:
                    cells.append("b" if color == Player.black else "w")

        last = game.last_move
        if last is None:
            last_move = None
        elif last.is_play:
            last_move = [last.point.row - 1, last.point.col - 1]
        else:
            last_move = "pass" if last.is_pass else "resign"

        score = game.result()
        winner = result = None
        if game.is_over():
            if last.is_resign:
                winner = game.next_player.name
                result = f"{winner[0].upper()}+R"
            else:
                winner = score.winner.name if score.winner else "draw"
                result = str(score)

        move_number = 0
        node = game
        while node.previous_state is not None:
            move_number += 1
            node = node.previous_state

        return {
            "type": "state",
            "size": size,
            "board": "".join(cells),
            "next": game.next_player.name,
            "last_move": last_move,
            "move_number": move_number,
            "score": {
                "black": score.black_score,
                "white": score.white_score,
                "komi": score.komi,
            },
            "players": {
                p.name: "human" if self.bot_for(p) is None else self.bot_names[p]
                for p in (Player.black, Player.white)
            },
            "over": game.is_over(),
            "winner": winner,
            "result": result,
            "thinking": thinking,
            "paused": self.paused,
        }

    async def new_game(self, msg):
        size = int(msg.get("size", 19))
        if size not in BOARD_SIZES:
            return await self.error(f"board size must be one of {BOARD_SIZES}")

        players, names = {}, {}
        for player in (Player.black, Player.white):
            name = msg.get(player.name, "human")
            if name == "human":
                players[player] = None
            elif name in bots:
                required = bot_board_size(bots[name])
                if required is not None and required != size:
                    return await self.error(f"{name} only plays {required}x{required}")
                players[player] = bots[name]
            else:
                return await self.error(f"unknown bot: {name}")
            names[player] = name

        self.stop_bots()
        self.players, self.bot_names = players, names
        self.paused = False
        self.game = GameState.new_game(size)
        await self.send(self.state())
        self.start_bots()

    async def human_move(self, move):
        game = self.game
        if game is None or not self.is_human_turn():
            return await self.error("not your turn")
        if not game.is_valid_move(move):
            return await self.error("illegal move")
        self.game = game.apply_move(move)
        await self.send(self.state())
        self.start_bots()

    async def undo(self):
        if self.game is None or self.game.previous_state is None:
            return
        self.stop_bots()
        game = self.game.previous_state
        assert game is not None
        if any(bot is None for bot in self.players.values()):
            while game.previous_state is not None and self.bot_for(game.next_player):
                game = game.previous_state
        else:
            self.paused = True
        self.game = game
        await self.send(self.state())
        self.start_bots()

    async def toggle_pause(self):
        if self.game is None:
            return
        self.paused = not self.paused
        if self.paused:
            self.stop_bots()
        await self.send(self.state())
        self.start_bots()

    async def handle(self, msg):
        kind = msg.get("type")
        if kind == "new":
            await self.new_game(msg)
        elif kind == "play":
            row, col = int(msg["row"]), int(msg["col"])
            await self.human_move(Move.play(Point(row + 1, col + 1)))
        elif kind == "pass":
            await self.human_move(Move.pass_turn())
        elif kind == "resign":
            await self.human_move(Move.resign())
        elif kind == "undo":
            await self.undo()
        elif kind == "pause":
            await self.toggle_pause()
        else:
            await self.error(f"unknown message type: {kind}")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session(ws)
    await session.send(
        {
            "type": "bots",
            "bots": [
                {"name": name, "board_size": bot_board_size(bot), "info": bot_info(bot)}
                for name, bot in bots.items()
            ],
        }
    )
    try:
        while True:
            await session.handle(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        session.stop_bots()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        help="defaults to the best supervised run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint is None:
        args.checkpoint = best_checkpoint()
    if args.checkpoint is not None and args.checkpoint.exists():
        bots["policy"] = load_policy_bot(args.checkpoint, device)
        print(f"loaded policy bot on {device}: {bots['policy'].info}")
    else:
        print(f"no checkpoint at {args.checkpoint}, only the random bot is available")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
