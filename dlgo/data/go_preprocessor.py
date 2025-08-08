import os
import numpy as np
from tqdm import tqdm
import pickle
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from dlgo.gotypes import Point
from dlgo.goboard import GameState, Move
from dlgo.data.process_files import extract_moves, setup_handicap_game


class GoGamePreprocessor:
    def __init__(
        self,
        encoder,
        data_directory="go_data/go_games/",
        output_directory="processed_go_data",
    ):
        self.encoder = encoder
        self.data_dir = os.path.join(os.getcwd(), data_directory)
        self.output_dir = os.path.join(os.getcwd(), output_directory)
        os.makedirs(self.output_dir, exist_ok=True)

    def _process_single_game(self, game_data):
        """Process a single game and return all board states and moves"""
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
            move_labels = []

            for move_idx in range(num_moves):
                board_tensor = self.encoder.encode(game)

                color, (row, col) = moves[move_idx]
                point = Point(row, col)

                move_index = self.encoder.encode_point(point)
                label = np.zeros(self.encoder.num_points())
                label[move_index] = 1

                boards.append(board_tensor.astype(np.float32))
                move_labels.append(label.astype(np.float32))

                move_obj = Move.play(point)
                game = game.apply_move(move_obj)

            return {
                "game_idx": game_idx,
                "boards": np.array(boards),
                "moves": np.array(move_labels),
                "num_moves": num_moves,
            }

        except Exception as e:
            print(f"Error processing game {game_idx}: {e}")
            return None

    def _save_game_npz(self, game_data, game_counter):
        """save a single game's data as NPZ file"""
        if game_data is None:
            return None

        filename = f"game_{game_counter:08d}.npz"
        filepath = os.path.join(self.output_dir, filename)

        try:
            np.savez_compressed(
                filepath,
                boards=game_data["boards"],
                moves=game_data["moves"],
                game_idx=game_data["game_idx"],
                num_moves=game_data["num_moves"],
            )
            return {
                "filename": filename,
                "filepath": filepath,
                "game_idx": game_data["game_idx"],
                "num_moves": game_data["num_moves"],
                "file_size": os.path.getsize(filepath),
            }
        except Exception as e:
            print(f"Error saving game {game_data['game_idx']}: {e}")
            return None

    def preprocess_games_parallel(
        self,
        max_moves_per_game=None,
        num_processes=None,
        batch_size=1000,
    ):
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 8)

        print("Counting games...")
        game_files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]

        saved_games = []
        total_samples = 0
        game_counter = 0

        for file_idx, file_name in enumerate(game_files):
            file_path = os.path.join(self.data_dir, file_name)
            print(
                f"""\nProcessing file {file_idx + 1}/{len(game_files)}: {file_name}"""
            )

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            games = content.split("\n")
            games = [game for game in games if game.strip()]
            print(f"  Found {len(games):,} games in file")

            for batch_start in range(0, len(games), batch_size):
                batch_end = min(batch_start + batch_size, len(games))
                game_batch = games[batch_start:batch_end]

                print(f"""  Processing batch: games {batch_start}-{batch_end - 1}""")

                batch_results = self._process_game_batch(
                    game_batch, batch_start, max_moves_per_game, num_processes
                )

                for game_data in batch_results:
                    if game_data is not None:
                        saved_info = self._save_game_npz(game_data, game_counter)
                        if saved_info is not None:
                            saved_games.append(saved_info)
                            total_samples += game_data["num_moves"]
                            game_counter += 1

                print(
                    f"""  Batch complete. Processed {len(batch_results)} games. """
                    f"""Total games saved: {game_counter}, Total samples: {
                        total_samples:,}"""
                )

                del batch_results
                del game_batch

        metadata = self._save_metadata(saved_games, total_samples)

        print(f"\nPreprocessing complete!")
        print(f"Games saved: {len(saved_games):,}")
        print(f"Total samples: {total_samples:,}")
        print(f"""Average moves per game: {total_samples / len(saved_games):.1f}""")

        return metadata

    def _process_game_batch(
        self, game_batch, batch_offset, max_moves_per_game, num_processes
    ):
        batch_data = [
            (batch_offset + i, game, max_moves_per_game)
            for i, game in enumerate(game_batch)
        ]

        batch_results = []

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # submit batch for processing
            future_to_game = {
                executor.submit(self._process_single_game, data): data[0]
                for data in batch_data
            }

            # collect results as they complete
            for future in tqdm(
                as_completed(future_to_game),
                total=len(batch_data),
                desc="  Processing batch",
            ):
                game_idx = future_to_game[future]
                try:
                    game_data = future.result()
                    batch_results.append(game_data)
                except Exception as e:
                    print(f"    Game {game_idx} failed: {e}")
                    batch_results.append(None)

        return batch_results

    def _save_metadata(self, saved_games, total_samples):
        metadata = {
            "total_games": len(saved_games),
            "total_samples": total_samples,
            "encoder_name": self.encoder.__class__.__name__,
            "board_shape": None,
            "move_shape": None,
            "games": saved_games,
        }

        if saved_games:
            first_game_path = saved_games[0]["filepath"]
            try:
                with np.load(first_game_path) as data:
                    # remove batch dim
                    metadata["board_shape"] = data["boards"].shape[1:]
                    metadata["move_shape"] = data["moves"].shape[1:]
            except Exception as e:
                print(f"Warning: Could not read shapes from first game: {e}")

        metadata_file = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            # convert numpy types to regular Python types for JSON serialization
            json_metadata = {
                "total_games": int(metadata["total_games"]),
                "total_samples": int(metadata["total_samples"]),
                "encoder_name": metadata["encoder_name"],
                "board_shape": [int(x) for x in metadata["board_shape"]]
                if metadata["board_shape"] is not None
                else None,
                "move_shape": [int(x) for x in metadata["move_shape"]]
                if metadata["move_shape"] is not None
                else None,
                "games": [
                    {
                        "filename": game["filename"],
                        "game_idx": int(game["game_idx"]),
                        "num_moves": int(game["num_moves"]),
                        "file_size": int(game["file_size"]),
                    }
                    for game in metadata["games"]
                ],
            }
            json.dump(json_metadata, f, indent=2)

        pickle_metadata_file = os.path.join(self.output_dir, "metadata.pkl")
        with open(pickle_metadata_file, "wb") as f:
            pickle.dump(metadata, f)

        print(f"Metadata saved to: {metadata_file}")
        return metadata

    def get_preprocessing_stats(self):
        if not os.path.exists(self.output_dir):
            return {"status": "No processed data found"}

        metadata_file = os.path.join(self.output_dir, "metadata.json")

        if not os.path.exists(metadata_file):
            npz_files = [f for f in os.listdir(self.output_dir) if f.endswith(".npz")]
            if npz_files:
                total_size = sum(
                    os.path.getsize(os.path.join(self.output_dir, f)) for f in npz_files
                )
                return {
                    "status": "Processed data found (no metadata)",
                    "total_games": len(npz_files),
                    "total_size_gb": total_size / (1024**3),
                    "format": "npz_files",
                }
            else:
                return {"status": "No processed data found"}

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            total_size = sum(game["file_size"] for game in metadata["games"])

            return {
                "status": "Complete",
                "format": "npz_files",
                "total_games": metadata["total_games"],
                "total_samples": metadata["total_samples"],
                "board_shape": metadata["board_shape"],
                "move_shape": metadata["move_shape"],
                "total_size_gb": total_size / (1024**3),
                "avg_moves_per_game": metadata["total_samples"]
                / metadata["total_games"]
                if metadata["total_games"] > 0
                else 0,
                "encoder_name": metadata["encoder_name"],
            }

        except Exception as e:
            return {"error": f"Error reading metadata: {e}"}

    def load_game_data(self, game_filename):
        filepath = os.path.join(self.output_dir, game_filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Game file not found: {filepath}")

        try:
            with np.load(filepath) as data:
                return {
                    "boards": data["boards"],
                    "moves": data["moves"],
                    "game_idx": int(data["game_idx"]),
                    "num_moves": int(data["num_moves"]),
                }
        except Exception as e:
            raise Exception(f"Error loading game file {game_filename}: {e}")

    def load_random_games(self, num_games=10):
        metadata_file = os.path.join(self.output_dir, "metadata.json")

        if not os.path.exists(metadata_file):
            raise FileNotFoundError("Metadata file not found. Run preprocessing first.")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        if len(metadata["games"]) < num_games:
            num_games = len(metadata["games"])

        import random

        selected_games = random.sample(metadata["games"], num_games)

        game_data = []
        for game_info in selected_games:
            try:
                data = self.load_game_data(game_info["filename"])
                game_data.append(data)
            except Exception as e:
                print(f"Warning: Could not load game {game_info['filename']}: {e}")

        return game_data

    def create_data_generator(self, batch_size=32, shuffle=True):
        metadata_file = os.path.join(self.output_dir, "metadata.json")

        if not os.path.exists(metadata_file):
            raise FileNotFoundError("Metadata file not found. Run preprocessing first.")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        games = metadata["games"]

        if shuffle:
            import random

            games = games.copy()
            random.shuffle(games)

        all_boards = []
        all_moves = []

        for game_info in games:
            try:
                game_data = self.load_game_data(game_info["filename"])
                all_boards.append(game_data["boards"])
                all_moves.append(game_data["moves"])
            except Exception as e:
                print(f"Warning: Skipping game {game_info['filename']}: {e}")
                continue

        if not all_boards:
            raise Exception("No valid game data found")

        all_boards = np.concatenate(all_boards, axis=0)
        all_moves = np.concatenate(all_moves, axis=0)

        if shuffle:
            # shuffle  combined data
            indices = np.random.permutation(len(all_boards))
            all_boards = all_boards[indices]
            all_moves = all_moves[indices]

        for i in range(0, len(all_boards), batch_size):
            end_idx = min(i + batch_size, len(all_boards))
            yield all_boards[i:end_idx], all_moves[i:end_idx]

    def _count_total_games(self):
        total_games = 0
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                games = [game for game in content.split("\n") if game.strip()]
                total_games += len(games)
        return total_games

    def preprocess_and_save(
        self,
        max_moves_per_game=None,
        num_processes=None,
        batch_size=1000,
    ):
        total_games = self._count_total_games()
        print(f"Total games to process: {total_games:,}")

        result = self.preprocess_games_parallel(
            max_moves_per_game=max_moves_per_game,
            num_processes=num_processes,
            batch_size=batch_size,
        )
        return result

    def cleanup_temp_files(self):
        temp_patterns = ["intermediate_results_", "temp_", ".tmp"]

        for filename in os.listdir(self.output_dir):
            if any(pattern in filename for pattern in temp_patterns):
                try:
                    os.remove(os.path.join(self.output_dir, filename))
                    print(f"Removed temp file: {filename}")
                except Exception as e:
                    print(f"Could not remove {filename}: {e}")


def process_single_game_worker(game_data):
    game_idx, sgf_content, max_moves_per_game, encoder = game_data

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
        move_labels = []

        for move_idx in range(num_moves):
            board_tensor = encoder.encode(game)

            color, (row, col) = moves[move_idx]
            point = Point(row, col)

            move_index = encoder.encode_point(point)
            label = np.zeros(encoder.num_points())
            label[move_index] = 1

            boards.append(board_tensor.astype(np.float32))
            move_labels.append(label.astype(np.float32))

            move_obj = Move.play(point)
            game = game.apply_move(move_obj)

        return {
            "game_idx": game_idx,
            "boards": np.array(boards),
            "moves": np.array(move_labels),
            "num_moves": num_moves,
        }

    except Exception as e:
        print(f"Error processing game {game_idx}: {e}")
        return None


if __name__ == "__main__":
    from dlgo.encoders.base import get_encoder_by_name

    encoder = get_encoder_by_name("oneplane", 19)
    preprocessor = GoGamePreprocessor(encoder)

    result = preprocessor.preprocess_and_save(
        max_moves_per_game=500,
        num_processes=4,
        batch_size=1000,
    )

    print(f"Preprocessing complete! Metadata: {result}")

    stats = preprocessor.get_preprocessing_stats()
    print(f"Stats: {stats}")
