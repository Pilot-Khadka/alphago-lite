import os
import json
import warnings
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from dlgo.gotypes import Point
from dlgo.goboard import GameState, Move
from dlgo.data.process_files import extract_moves, setup_handicap_game


class GoGamePreprocessor:
    def __init__(
        self,
        encoder,
        data_directory="go_data/go_games/",
        output_directory="processed_go_data",
        shard_size=200000,  # positions per shard
        use_float16=True,  # reduce memory usage
        bitpack_boards=False,  # experimental: pack binary planes to bits
    ):
        self.encoder = encoder
        self.data_dir = os.path.join(os.getcwd(), data_directory)
        self.output_dir = os.path.join(os.getcwd(), output_directory)
        self.shard_size = shard_size
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

    def preprocess_to_shards(
        self,
        max_moves_per_game=None,
        num_processes=None,
        batch_size=1000,
    ):
        """save as memory-mapped shards"""
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 8)

        print("Starting shard-based preprocessing...")

        # collect all games first to estimate shard structure
        all_games = self._collect_all_games()
        total_games = len(all_games)
        print(f"Found {total_games:,} games total")

        current_shard_boards = []
        current_shard_moves = []
        current_shard_positions = 0
        shard_id = 0
        shard_metadata = []
        total_positions = 0

        print("Processing games and building shards...")

        for batch_start in range(0, total_games, batch_size):
            batch_end = min(batch_start + batch_size, total_games)
            game_batch = all_games[batch_start:batch_end]

            print(f"Processing batch: games {batch_start}-{batch_end - 1}")

            batch_results = self._process_game_batch(
                game_batch, batch_start, max_moves_per_game, num_processes
            )

            for game_data in batch_results:
                if game_data is None:
                    continue

                game_positions = len(game_data["boards"])

                # check if adding this game would exceed shard size
                if (
                    len(current_shard_boards) + game_positions > self.shard_size
                    and len(current_shard_boards) > 0
                ):
                    self._save_shard(
                        shard_id,
                        current_shard_boards,
                        current_shard_moves,
                        shard_metadata,
                    )
                    shard_id += 1
                    current_shard_boards = []
                    current_shard_moves = []

                if self.bitpack_boards:
                    for b in game_data["boards"]:
                        if not isinstance(b, dict):
                            raise TypeError(
                                f"Expected dict in boards but got {type(b)}"
                            )
                    current_shard_boards.extend(game_data["boards"])
                else:
                    if isinstance(game_data["boards"], np.ndarray):
                        current_shard_boards.extend(game_data["boards"])
                    else:
                        current_shard_boards.extend(list(game_data["boards"]))

                current_shard_moves.extend(game_data["moves"])
                total_positions += game_positions

            print(f"""Batch complete. Total positions so far: {total_positions:,}""")

        # save final shard if it has data
        if current_shard_positions > 0:
            self._save_shard(
                shard_id, current_shard_boards, current_shard_moves, shard_metadata
            )

        self._create_global_index(shard_metadata, total_positions)

        print("Shard-based preprocessing complete!")
        print(f"Total shards: {len(shard_metadata)}")
        print(f"Total positions: {total_positions:,}")
        print(
            f"""Average positions per shard: {
                total_positions / len(shard_metadata):.0f}"""
        )

        return {
            "total_shards": len(shard_metadata),
            "total_positions": total_positions,
            "shard_metadata": shard_metadata,
        }

    def _collect_all_games(self):
        all_games = []
        game_files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]

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

    def _save_shard(self, shard_id, boards, moves, shard_metadata):
        if self.bitpack_boards:
            packed_data_list = [b["packed_data"] for b in boards]
            max_len = max(len(arr) for arr in packed_data_list)

            padded_boards = np.zeros((len(packed_data_list), max_len), dtype=np.uint8)
            for i, arr in enumerate(packed_data_list):
                padded_boards[i, : len(arr)] = arr

            boards_array = padded_boards
        else:
            # all boards have the same shape and create proper 2D array
            if not boards:
                raise ValueError("No boards to save in shard")

            # get expected shape from first board
            first_board = boards[0]
            expected_shape = first_board.shape

            # check all boards have the same shape
            for i, board in enumerate(boards):
                if board.shape != expected_shape:
                    raise ValueError(
                        f"Board {i} has shape {board.shape}, expected {expected_shape}"
                    )

            num_boards = len(boards)
            boards_array = np.empty(
                (num_boards, *expected_shape), dtype=self.board_dtype
            )

            for i, board in enumerate(boards):
                boards_array[i] = board.astype(self.board_dtype)

        moves_array = np.array(moves, dtype=self.move_dtype)

        boards_file = os.path.join(self.output_dir, f"boards_shard_{shard_id:03d}.dat")
        moves_file = os.path.join(self.output_dir, f"moves_shard_{shard_id:03d}.dat")

        boards_array.tofile(boards_file)
        moves_array.tofile(moves_file)

        shard_info = {
            "shard_id": shard_id,
            "num_positions": len(boards_array),
            "boards_file": f"boards_shard_{shard_id:03d}.dat",
            "moves_file": f"moves_shard_{shard_id:03d}.dat",
            # convert to list for JSON
            "boards_shape": list(boards_array.shape),
            "moves_shape": list(moves_array.shape),
            "boards_dtype": str(boards_array.dtype),
            "moves_dtype": str(moves_array.dtype),
        }

        shard_metadata.append(shard_info)
        print(f"Saved shard {shard_id}: {len(boards_array):,} positions")

    def _create_global_index(self, shard_metadata, total_positions):
        """Create global index mapping position_id -> (shard_id, local_offset)
        Validates shard sizes from actual .dat files before building index.
        """

        actual_total_positions = 0

        for shard_info in shard_metadata:
            boards_file = os.path.join(self.output_dir, shard_info["boards_file"])
            if not os.path.exists(boards_file):
                raise FileNotFoundError(f"Missing boards file: {boards_file}")

            dtype_size = np.dtype(shard_info["boards_dtype"]).itemsize
            board_size = np.prod(shard_info["boards_shape"][1:])
            file_size = os.path.getsize(boards_file)

            actual_positions = file_size // (dtype_size * board_size)
            if actual_positions != shard_info["num_positions"]:
                print(
                    f"Shard {shard_info['shard_id']} num_positions mismatch: "
                    f"metadata={shard_info['num_positions']}, actual={actual_positions}"
                )
                shard_info["num_positions"] = actual_positions

            actual_total_positions += actual_positions

        global_to_shard = np.zeros(actual_total_positions, dtype=np.uint16)
        global_to_offset = np.zeros(actual_total_positions, dtype=np.uint32)

        pos = 0
        for shard_info in shard_metadata:
            shard_id = shard_info["shard_id"]
            num_pos = shard_info["num_positions"]

            start_pos = pos
            end_pos = pos + num_pos

            global_to_shard[start_pos:end_pos] = shard_id
            global_to_offset[start_pos:end_pos] = np.arange(num_pos)

            shard_info["start_position"] = start_pos
            shard_info["end_position"] = end_pos

            pos = end_pos

        # save index files without numpy headers (raw binary)
        index_shard_file = os.path.join(self.output_dir, "global_index_shard.npy")
        index_offset_file = os.path.join(self.output_dir, "global_index_offset.npy")
        global_to_shard.tofile(index_shard_file)
        global_to_offset.tofile(index_offset_file)

        individual_board_shape = None
        if shard_metadata:
            first_shard_boards_shape = shard_metadata[0]["boards_shape"]
            if len(first_shard_boards_shape) > 1:
                individual_board_shape = list(first_shard_boards_shape[1:])
            else:
                try:
                    individual_board_shape = [self.encoder.num_points]
                except Exception as e:
                    warnings.warn(
                        f"Could not get num_points from encoder: {
                            e
                        }. Using default 361 for 19x19 board."
                    )
                    individual_board_shape = [361]  # Default for 19x19

        metadata = {
            "total_positions": actual_total_positions,
            "total_shards": len(shard_metadata),
            "shard_size": self.shard_size,
            "encoder_name": self.encoder.__class__.__name__,
            "board_dtype": str(self.board_dtype),
            "move_dtype": str(self.move_dtype),
            "use_float16": self.use_float16,
            "bitpack_boards": self.bitpack_boards,
            "individual_board_shape": individual_board_shape,
            "shards": shard_metadata,
            "index_files": {
                "shard_mapping": "global_index_shard.npy",
                "offset_mapping": "global_index_offset.npy",
            },
        }

        metadata_file = os.path.join(self.output_dir, "shard_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(self._convert_numpy_types(metadata), f, indent=2)

        print(f"✅ Global index saved. Metadata: {metadata_file}")
        print(f"Total positions: {actual_total_positions}")
        print(f"Total shards: {len(shard_metadata)}")
        print(f"Individual board shape: {individual_board_shape}")

    def _convert_numpy_types(self, obj):
        """Convert numpy types to regular Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, tuple):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def get_preprocessing_stats(self):
        metadata_file = os.path.join(self.output_dir, "shard_metadata.json")

        if not os.path.exists(metadata_file):
            return {"status": "No shard-based data found"}

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            total_size = 0
            for shard in metadata["shards"]:
                boards_file = os.path.join(self.output_dir, shard["boards_file"])
                moves_file = os.path.join(self.output_dir, shard["moves_file"])
                if os.path.exists(boards_file):
                    total_size += os.path.getsize(boards_file)
                if os.path.exists(moves_file):
                    total_size += os.path.getsize(moves_file)

            return {
                "status": "Complete",
                "format": "memory_mapped_shards",
                "total_shards": metadata["total_shards"],
                "total_positions": metadata["total_positions"],
                "individual_board_shape": metadata["individual_board_shape"],
                "board_dtype": metadata["board_dtype"],
                "move_dtype": metadata["move_dtype"],
                "total_size_gb": total_size / (1024**3),
                "avg_positions_per_shard": metadata["total_positions"]
                / metadata["total_shards"],
                "encoder_name": metadata["encoder_name"],
                "use_float16": metadata.get("use_float16", False),
                "bitpack_boards": metadata.get("bitpack_boards", False),
            }

        except Exception as e:
            return {"error": f"Error reading shard metadata: {e}"}
