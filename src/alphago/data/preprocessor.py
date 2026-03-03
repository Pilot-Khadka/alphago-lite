import os
import json
import zipfile
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..go.gotypes import Point
from ..encoders.base import Encoder
from ..go.goboard import GameState, Move
from ..data.file_utils import extract_moves, setup_handicap_game


CHECKPOINT_FILE = "checkpoint.json"
SHARDS_PER_ZIP = 1


class GoGamePreprocessor:
    def __init__(
        self,
        encoder: Encoder,
        data_directory: str,
        output_directory: str,
        use_float16: bool = True,
        board_size: int = 19,
        use_tqdm: bool = False,
    ):
        if not os.path.exists(data_directory):
            raise RuntimeError(f"path {data_directory} does not exists")

        self.board_size = board_size
        self.encoder = encoder
        self.data_dir = data_directory
        self.output_dir = output_directory
        self.use_float16 = use_float16
        self.use_tqdm = use_tqdm

        os.makedirs(self.output_dir, exist_ok=True)

        self.board_dtype = np.float16 if use_float16 else np.float32
        self.move_dtype = np.uint16

    def _checkpoint_path(self):
        return os.path.join(self.output_dir, CHECKPOINT_FILE)

    def _load_checkpoint(self) -> int:
        path = self._checkpoint_path()
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)["next_shard_game_offset"]
        return 0

    def _save_checkpoint(self, next_shard_game_offset: int) -> None:
        with open(self._checkpoint_path(), "w") as f:
            json.dump({"next_shard_game_offset": next_shard_game_offset}, f)

    def _shard_npy_paths(self, shard_id: int) -> tuple[str, str]:
        prefix = f"shard_{shard_id:04d}_"
        boards = os.path.join(self.output_dir, f"{prefix}boards.npy")
        moves = os.path.join(self.output_dir, f"{prefix}moves.npy")
        return boards, moves

    def _zip_shard_group(self, shard_ids: list[int]) -> None:
        first, last = shard_ids[0], shard_ids[-1]
        zip_name = f"shards_{first:04d}-{last:04d}.zip"
        zip_path = os.path.join(self.output_dir, zip_name)

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for shard_id in shard_ids:
                for npy_path in self._shard_npy_paths(shard_id):
                    if os.path.exists(npy_path):
                        zf.write(npy_path, arcname=os.path.basename(npy_path))
                        os.remove(npy_path)

        print(f"Compressed shards {first}-{last} → {zip_name}")

    def preprocess(
        self,
        all_games: list[str],
        games_per_shard: int = 5000,
        max_moves_per_game: int | None = None,
        num_processes: int = 4,
    ) -> None:
        resume_offset = self._load_checkpoint()
        if resume_offset > 0:
            print(
                f"Resuming from game {resume_offset} (skipping {resume_offset} already-processed games)"
            )

        shard_id = resume_offset // games_per_shard
        total_games = len(all_games)

        print(
            f"Processing {total_games - resume_offset} remaining games ({total_games} total)..."
        )

        # Tracks shard IDs that have been written but not yet zipped.
        pending_shard_ids: list[int] = []

        for offset in range(resume_offset, total_games, games_per_shard):
            game_slice = all_games[offset : offset + games_per_shard]

            results = self._process_game_batch(
                game_slice,
                batch_offset=offset,
                max_moves_per_game=max_moves_per_game,
                num_processes=num_processes,
            )

            boards_buf: list[np.ndarray] = []
            moves_buf: list[np.ndarray] = []

            for result in results:
                if result is None:
                    continue

                # pyrefly: ignore
                boards_buf.extend(result["boards"])
                # pyrefly: ignore
                moves_buf.append(result["moves"])

            if boards_buf:
                self._flush_shard(boards_buf, moves_buf, shard_id)
                pending_shard_ids.append(shard_id)

                if len(pending_shard_ids) == SHARDS_PER_ZIP:
                    self._zip_shard_group(pending_shard_ids)
                    pending_shard_ids = []

            self._save_checkpoint(offset + len(game_slice))
            shard_id += 1

        if pending_shard_ids:
            self._zip_shard_group(pending_shard_ids)

        print("Done.")

    def _flush_shard(self, boards_buf, moves_buf, shard_id):
        boards = np.stack(boards_buf, axis=0)
        moves = np.concatenate(moves_buf, axis=0)

        perm = np.random.permutation(len(moves))
        boards = boards[perm]
        moves = moves[perm]

        self._write_shard(boards, moves, f"shard_{shard_id:04d}_")
        print(
            f"Saved shard {shard_id}: {len(moves)} positions from {len(moves_buf)} games"
        )

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
        total = len(batch_data)

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_game = {
                executor.submit(self._process_single_game, data): data[0]
                for data in batch_data
            }

            if self.use_tqdm:
                iterator = tqdm(
                    as_completed(future_to_game),
                    total=total,
                    desc="Processing games",
                )
            else:
                iterator = as_completed(future_to_game)

            completed = 0
            for future in iterator:
                game_idx = future_to_game[future]
                try:
                    batch_results.append(future.result())
                except Exception as e:
                    print(f"Game {game_idx} failed: {e}")
                    batch_results.append(None)

                if not self.use_tqdm:
                    completed += 1
                    if completed % 100 == 0:
                        print(f"Completed {completed}/{total} games", end="\r")

        if not self.use_tqdm:
            print()

        return batch_results
