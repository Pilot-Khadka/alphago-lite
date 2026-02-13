import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..gotypes import Point
from ..encoders.base import Encoder
from ..goboard import GameState, Move
from ..data.file_utils import extract_moves, setup_handicap_game


class GoGamePreprocessor:
    def __init__(
        self,
        encoder: Encoder,
        data_directory: str,
        output_directory: str,
        use_float16: bool = True,
        board_size: int = 19,
    ):
        if not os.path.exists(data_directory):
            raise RuntimeError(f"path {data_directory} does not exists")

        self.board_size = board_size
        self.encoder = encoder
        self.data_dir = data_directory
        self.output_dir = output_directory
        self.use_float16 = use_float16

        os.makedirs(self.output_dir, exist_ok=True)

        self.board_dtype = np.float16 if use_float16 else np.float32
        self.move_dtype = np.uint16

    def preprocess(
        self,
        all_games: list[str],
        max_moves_per_game: int | None = None,
        num_processes: int = 4,
        shard_size: int = 100_000,
    ) -> None:
        print(f"Processing {len(all_games)} games...")

        shard_id = 0
        boards_buf: list[np.ndarray] = []
        moves_buf: list[np.ndarray] = []

        batch_size = 5000
        offset = 0

        while offset < len(all_games):
            batch = all_games[offset : offset + batch_size]
            offset += batch_size

            results = self._process_game_batch(
                batch,
                batch_offset=offset - batch_size,
                max_moves_per_game=max_moves_per_game,
                num_processes=num_processes,
            )

            for result in results:
                if result is None:
                    continue

                # pyrefly: ignore [bad-argument-type]
                boards_buf.extend(result["boards"])
                # pyrefly: ignore [bad-argument-type]
                moves_buf.append(result["moves"])

                if len(boards_buf) >= shard_size:
                    self._flush_shard(boards_buf, moves_buf, shard_id)
                    shard_id += 1
                    boards_buf.clear()
                    moves_buf.clear()

        if boards_buf:
            self._flush_shard(boards_buf, moves_buf, shard_id)

        print("Done.")

    def _flush_shard(self, boards_buf, moves_buf, shard_id):
        boards = np.stack(boards_buf, axis=0)
        moves = np.concatenate(moves_buf, axis=0)

        perm = np.random.permutation(len(moves))
        boards = boards[perm]
        moves = moves[perm]

        prefix = f"shard_{shard_id:04d}_"
        self._write_shard(boards, moves, prefix)

        print(f"Saved shard {shard_id}: {len(moves)} positions")

    def _write_shard(self, boards: np.ndarray, moves: np.ndarray, prefix: str) -> None:
        np.save(os.path.join(self.output_dir, f"{prefix}boards.npy"), boards)
        np.save(os.path.join(self.output_dir, f"{prefix}moves.npy"), moves)

    def _process_single_game(self, game_data):
        game_idx, sgf_content, max_moves_per_game = game_data

        try:
            game = GameState.new_game(self.board_size)
            moves, handicap_info = extract_moves(sgf_content)

            if handicap_info["is_handicap_game"]:
                game, _ = setup_handicap_game(self.board_size, handicap_info)

            num_moves = len(moves)
            if max_moves_per_game:
                num_moves = min(num_moves, max_moves_per_game)

            if num_moves == 0:
                return None

            boards = []
            move_indices = []
            expected_shape = None

            for move_idx in range(num_moves):
                board_tensor = self.encoder.encode(game)

                if expected_shape is None:
                    expected_shape = board_tensor.shape
                elif board_tensor.shape != expected_shape:
                    print(
                        f"Warning: Game {game_idx}, move {move_idx}: "
                        f"board shape {board_tensor.shape} != expected {expected_shape}"
                    )
                    return None

                # pyrefly: ignore [not-iterable]
                color, (row, col) = moves[move_idx]
                point = Point(row, col)

                boards.append(board_tensor.astype(self.board_dtype).copy())
                move_indices.append(self.encoder.encode_point(point))

                game = game.apply_move(Move.play(point))

            return {
                "game_idx": game_idx,
                "boards": boards,
                "moves": np.array(move_indices, dtype=self.move_dtype),
                "num_moves": num_moves,
            }

        except Exception as e:
            print(f"Error processing game {game_idx}: {e}")
            return None

    def collect_all_games(self) -> list[str]:
        all_games = []
        game_files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]

        for file_idx, file_name in enumerate(game_files):
            file_path = os.path.join(self.data_dir, file_name)
            print(f"Reading file {file_idx + 1}/{len(game_files)}: {file_name}")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            games = [g.strip() for g in content.split("\n") if g.strip()]
            all_games.extend(games)

        return all_games

    def _process_game_batch(
        self,
        game_batch,
        batch_offset,
        max_moves_per_game,
        num_processes,
    ):
        batch_data = [
            (batch_offset + i, game, max_moves_per_game)
            for i, game in enumerate(game_batch)
        ]

        batch_results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_game = {
                executor.submit(self._process_single_game, data): data[0]
                for data in batch_data
            }

            for future in tqdm(
                as_completed(future_to_game),
                total=len(batch_data),
                desc="Processing games",
            ):
                game_idx = future_to_game[future]
                try:
                    batch_results.append(future.result())
                except Exception as e:
                    print(f"Game {game_idx} failed: {e}")
                    batch_results.append(None)

        return batch_results
