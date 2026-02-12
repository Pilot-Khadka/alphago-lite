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
        use_float16: bool = True,  # reduce memory usage
        bitpack_boards: bool = False,  # experimental: pack binary planes to bits
    ):
        self.encoder = encoder
        self.data_dir = os.path.join(os.getcwd(), data_directory)
        self.output_dir = os.path.join(os.getcwd(), output_directory)
        self.use_float16 = use_float16
        self.bitpack_boards = bitpack_boards

        os.makedirs(self.output_dir, exist_ok=True)

        self.board_dtype = (
            np.uint8 if bitpack_boards else (np.float16 if use_float16 else np.float32)
        )
        self.move_dtype = np.uint16  # store move indices instead of one-hot

    def _process_single_game(self, game_data):
        game_idx, sgf_content, max_moves_per_game = game_data

        try:
            board_size = 19
            game = GameState.new_game(board_size)

            moves, handicap_info = extract_moves(sgf_content)

            if handicap_info["is_handicap_game"]:
                game, handicap_moves = setup_handicap_game(board_size, handicap_info)

            num_moves = len(moves)
            if max_moves_per_game:
                num_moves = min(num_moves, max_moves_per_game)

            if num_moves == 0:
                return None

            boards = []
            move_indices = []

            for move_idx in range(num_moves):
                board_tensor = self.encoder.encode(game)

                if move_idx == 0:
                    expected_shape = board_tensor.shape
                elif board_tensor.shape != expected_shape:
                    print(
                        f"Warning: Game {game_idx}, move {move_idx}: "
                        f"board shape {board_tensor.shape} != expected {expected_shape}"
                    )
                    return None

                color, (row, col) = moves[move_idx]
                point = Point(row, col)
                move_index = self.encoder.encode_point(point)

                if self.bitpack_boards:
                    boards.append(self._pack_board_bits(board_tensor))
                else:
                    # Ensure consistent dtype and make a copy to avoid reference issues
                    board_copy = board_tensor.astype(self.board_dtype).copy()
                    boards.append(board_copy)

                move_indices.append(move_index)

                move_obj = Move.play(point)
                game = game.apply_move(move_obj)

            if not self.bitpack_boards:
                first_shape = boards[0].shape
                for i, board in enumerate(boards):
                    if board.shape != first_shape:
                        print(
                            f"Error: Game {game_idx}, inconsistent board shapes: "
                            f"board {i} has shape {board.shape}, expected {first_shape}"
                        )
                        return None

            return {
                "game_idx": game_idx,
                "boards": boards if self.bitpack_boards else boards,  # Keep as list
                "moves": np.array(move_indices, dtype=self.move_dtype),
                "num_moves": num_moves,
            }

        except Exception as e:
            print(f"Error processing game {game_idx}: {e}")
            return None

    def _pack_board_bits(self, board_tensor):
        if board_tensor.dtype != bool and not np.all(np.isin(board_tensor, [0, 1])):
            # not binary - fallback to regular encoding
            # instead of returning ndarray, pack as bytes anyway (or store raw data in 'packed_data')
            packed = board_tensor.astype(self.board_dtype).tobytes()
            original_shape = board_tensor.shape
            return {
                "packed_data": np.frombuffer(packed, dtype=np.uint8),
                "original_shape": original_shape,
            }

        # reshape to 2D and pack bits
        original_shape = board_tensor.shape
        flat_board = board_tensor.astype(bool).flatten()
        packed = np.packbits(flat_board)

        # original shape info for unpacking
        return {"packed_data": packed, "original_shape": original_shape}

    def _unpack_board_bits(self, packed_board_data):
        if isinstance(packed_board_data, dict):
            packed_data = packed_board_data["packed_data"]
            original_shape = packed_board_data["original_shape"]

            unpacked = np.unpackbits(packed_data)
            total_elements = np.prod(original_shape)
            unpacked = unpacked[:total_elements]  # Trim padding
            return unpacked.reshape(original_shape).astype(np.float32)
        else:
            # regular data, just convert dtype
            return packed_board_data.astype(np.float32)

    def _collect_all_games(self):
        all_games = []
        game_files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
        print("game files:", game_files)

        for file_idx, file_name in enumerate(game_files):
            file_path = os.path.join(self.data_dir, file_name)
            print(f"""Reading file {file_idx + 1}/{len(game_files)}: {file_name}""")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            games = content.split("\n")
            games = [game.strip() for game in games if game.strip()]
            all_games.extend(games)

        return all_games

    def _process_game_batch(
        self, game_batch, batch_offset, max_moves_per_game, num_processes
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
                desc="  Processing games",
            ):
                game_idx = future_to_game[future]
                try:
                    game_data = future.result()
                    batch_results.append(game_data)
                except Exception as e:
                    print(f"    Game {game_idx} failed: {e}")
                    batch_results.append(None)

        return batch_results
