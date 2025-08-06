import os
import numpy as np
from tqdm import tqdm
import pickle
import h5py
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
                game, handicap_moves = setup_handicap_game(
                    board_size, handicap_info)

            num_moves = len(moves)
            if max_moves_per_game:
                num_moves = min(num_moves, max_moves_per_game)

            game_samples = []

            for move_idx in range(num_moves):
                # Get current board state
                board_tensor = self.encoder.encode(game)

                # Get the move to be made
                color, (row, col) = moves[move_idx]
                point = Point(row, col)

                # Encode the move
                move_index = self.encoder.encode_point(point)
                label = np.zeros(self.encoder.num_points())
                label[move_index] = 1

                game_samples.append(
                    {
                        "board": board_tensor.astype(np.float32),
                        "move": label.astype(np.float32),
                        "game_idx": game_idx,
                        "move_idx": move_idx,
                    }
                )

                # Apply the move for next iteration
                move_obj = Move.play(point)
                game = game.apply_move(move_obj)

            return game_samples

        except Exception as e:
            print(f"Error processing game {game_idx}: {e}")
            return []

    def preprocess_games_parallel(
        self,
        max_moves_per_game=None,
        num_processes=None,
        batch_size=1000,
        output_filename="processed_games.h5",
    ):
        """Preprocess all games using multiprocessing with true memory-efficient batching"""
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 8)

        print("Counting games...")
        game_files = [f for f in os.listdir(
            self.data_dir) if f.endswith(".txt")]

        # Initialize HDF5 file for incremental writing
        output_path = os.path.join(self.output_dir, output_filename)
        h5_file = None
        datasets = {}
        total_samples_written = 0

        try:
            # Process each file separately to avoid loading all games into memory
            for file_idx, file_name in enumerate(game_files):
                file_path = os.path.join(self.data_dir, file_name)
                print(
                    f"""\nProcessing file {file_idx + 1}/{len(game_files)}: {
                        file_name
                    }"""
                )

                # Read games from current file in batches
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                games = content.split("\n")
                games = [game for game in games if game.strip()]
                print(f"  Found {len(games):,} games in file")

                # Process games in batches
                for batch_start in range(0, len(games), batch_size):
                    batch_end = min(batch_start + batch_size, len(games))
                    game_batch = games[batch_start:batch_end]

                    print(
                        f"""  Processing batch: games {
                            batch_start}-{batch_end - 1}"""
                    )

                    # Process this batch
                    batch_samples = self._process_game_batch(
                        game_batch, batch_start, max_moves_per_game, num_processes
                    )

                    if batch_samples:
                        # Initialize HDF5 file on first batch
                        if h5_file is None:
                            h5_file, datasets = self._initialize_hdf5_file(
                                output_path, batch_samples[0]
                            )

                        # Write batch to HDF5 file
                        self._write_batch_to_hdf5(
                            datasets, batch_samples, total_samples_written
                        )
                        total_samples_written += len(batch_samples)

                        print(
                            f"""  Batch complete. Wrote {
                                len(batch_samples):,} samples. Total: {
                                total_samples_written:,}"""
                        )

                    # Clear batch from memory immediately
                    del batch_samples
                    del game_batch

            # Update final dataset size and metadata
            if h5_file is not None:
                h5_file.attrs["num_samples"] = total_samples_written
                print(
                    f"""\nPreprocessing complete! Total samples: {
                        total_samples_written:,}"""
                )

        finally:
            # Always close the HDF5 file
            if h5_file is not None:
                h5_file.close()

        return output_path if total_samples_written > 0 else None

    def _process_game_batch(
        self, game_batch, batch_offset, max_moves_per_game, num_processes
    ):
        """Process a single batch of games and return samples"""
        # Prepare batch data for parallel processing
        batch_data = [
            (batch_offset + i, game, max_moves_per_game)
            for i, game in enumerate(game_batch)
        ]

        batch_samples = []

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit batch for processing
            future_to_game = {
                executor.submit(self._process_single_game, data): data[0]
                for data in batch_data
            }

            # Collect results as they complete
            for future in tqdm(
                as_completed(future_to_game),
                total=len(batch_data),
                desc="  Processing batch",
            ):
                game_idx = future_to_game[future]
                try:
                    game_samples = future.result()
                    batch_samples.extend(game_samples)
                except Exception as e:
                    print(f"    Game {game_idx} failed: {e}")

        return batch_samples

    def _initialize_hdf5_file(self, output_path, sample_data):
        """Initialize HDF5 file with resizable datasets"""
        print(f"Initializing HDF5 file: {output_path}")

        h5_file = h5py.File(output_path, "w")

        # Get shapes from sample
        board_shape = sample_data["board"].shape
        move_shape = sample_data["move"].shape

        # Create resizable datasets (start with small size, expand as needed)
        datasets = {
            "boards": h5_file.create_dataset(
                "boards",
                shape=(0,) + board_shape,
                maxshape=(None,) + board_shape,
                dtype=np.float32,
                compression="gzip",
                compression_opts=6,
                chunks=True,
            ),
            "moves": h5_file.create_dataset(
                "moves",
                shape=(0,) + move_shape,
                maxshape=(None,) + move_shape,
                dtype=np.float32,
                compression="gzip",
                compression_opts=6,
                chunks=True,
            ),
            "game_indices": h5_file.create_dataset(
                "game_indices",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=True,
            ),
            "move_indices": h5_file.create_dataset(
                "move_indices",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=True,
            ),
        }

        # Save metadata
        h5_file.attrs["board_shape"] = board_shape
        h5_file.attrs["move_shape"] = move_shape
        h5_file.attrs["encoder_name"] = self.encoder.__class__.__name__

        return h5_file, datasets

    def _write_batch_to_hdf5(self, datasets, batch_samples, start_index):
        """Write a batch of samples to HDF5 datasets"""
        if not batch_samples:
            return

        batch_size = len(batch_samples)
        end_index = start_index + batch_size

        # Resize datasets to accommodate new data
        for dataset_name in datasets:
            datasets[dataset_name].resize(
                (end_index,) + datasets[dataset_name].shape[1:]
            )

        # Write batch data
        for i, sample in enumerate(batch_samples):
            idx = start_index + i
            datasets["boards"][idx] = sample["board"]
            datasets["moves"][idx] = sample["move"]
            datasets["game_indices"][idx] = sample["game_idx"]
            datasets["move_indices"][idx] = sample["move_idx"]

    def get_preprocessing_stats(self):
        """Get statistics about processed data"""
        if not os.path.exists(self.output_dir):
            return {"status": "No processed data found"}

        # Check for single file
        single_files = [
            f
            for f in os.listdir(self.output_dir)
            if f.endswith(".h5") and not f.startswith("processed_chunk_")
        ]

        # Check for chunked files
        metadata_file = os.path.join(self.output_dir, "chunk_metadata.pkl")

        stats = {}

        if single_files:
            # Single file format
            try:
                filepath = os.path.join(self.output_dir, single_files[0])
                with h5py.File(filepath, "r") as f:
                    stats = {
                        "format": "single",
                        "file": single_files[0],
                        "total_samples": f.attrs.get("num_samples", len(f["boards"])),
                        "board_shape": f.attrs.get(
                            "board_shape", f["boards"].shape[1:]
                        ),
                        "file_size_gb": os.path.getsize(filepath) / (1024**3),
                    }
            except Exception as e:
                stats["error"] = f"Error reading single file: {e}"

        elif os.path.exists(metadata_file):
            # Chunked format
            try:
                with open(metadata_file, "rb") as f:
                    metadata = pickle.load(f)

                total_size = sum(
                    os.path.getsize(f)
                    for f in metadata["chunk_files"]
                    if os.path.exists(f)
                )

                stats = {
                    "format": "chunked",
                    "total_samples": metadata["total_samples"],
                    "num_chunks": metadata["num_chunks"],
                    "chunk_size": metadata["chunk_size"],
                    "total_size_gb": total_size / (1024**3),
                }
            except Exception as e:
                stats["error"] = f"Error reading chunk metadata: {e}"

        else:
            stats["status"] = "No processed data found"

        return stats

    def _save_intermediate_results(self, samples, games_processed):
        """Save intermediate results to prevent data loss"""
        filename = f"intermediate_results_{games_processed}_games.pkl"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, "wb") as f:
                pickle.dump(samples, f)
            print(
                f"""    Saved intermediate results: {len(samples):,} samples to {
                    filename
                }"""
            )
        except Exception as e:
            print(f"    Warning: Could not save intermediate results: {e}")

    def _count_total_games(self):
        """Count total number of games without loading them into memory"""
        total_games = 0
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                games = [game for game in content.split("\n") if game.strip()]
                total_games += len(games)
        return total_games

    def _read_games_from_file(self):
        """Read all SGF games from text files (kept for backward compatibility)"""
        print("Warning: _read_games_from_file loads all games into memory.")
        print("Consider using preprocess_games_parallel with batching instead.")

        all_games = []
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.data_dir, file_name)
                print(f"Reading: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                games = content.split("\n")
                games = [game for game in games if game.strip()]
                all_games.extend(games)

        print(f"Total games loaded: {len(all_games):,}")
        return all_games

    def preprocess_and_save(
        self,
        max_moves_per_game=None,
        save_format="single",
        chunk_size=10000,
        num_processes=None,
        batch_size=1000,
        output_filename="processed_games.h5",
    ):
        """Complete preprocessing pipeline with true memory-efficient processing"""
        # Get total game count for progress tracking
        total_games = self._count_total_games()
        print(f"Total games to process: {total_games:,}")

        if save_format == "single":
            # Direct streaming to single HDF5 file
            result = self.preprocess_games_parallel(
                max_moves_per_game=max_moves_per_game,
                num_processes=num_processes,
                batch_size=batch_size,
                output_filename=output_filename,
            )
            return result

        elif save_format == "chunks":
            # Process in chunks and save separate files
            return self.preprocess_and_save_chunks(
                max_moves_per_game=max_moves_per_game,
                num_processes=num_processes,
                batch_size=batch_size,
                chunk_size=chunk_size,
            )
        else:
            raise ValueError("save_format must be 'single' or 'chunks'")

    def preprocess_and_save_chunks(
        self,
        max_moves_per_game=None,
        num_processes=None,
        batch_size=1000,
        chunk_size=10000,
    ):
        """Process and save data in chunks without loading everything into memory"""
        if num_processes is None:
            num_processes = min(mp.cpu_count(), 8)

        game_files = [f for f in os.listdir(
            self.data_dir) if f.endswith(".txt")]

        chunk_files = []
        current_chunk_samples = []
        current_chunk_idx = 0
        total_samples = 0

        try:
            # Process each file
            for file_idx, file_name in enumerate(game_files):
                file_path = os.path.join(self.data_dir, file_name)
                print(
                    f"""\nProcessing file {file_idx + 1}/{len(game_files)}: {
                        file_name
                    }"""
                )

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                games = content.split("\n")
                games = [game for game in games if game.strip()]

                # Process games in batches
                for batch_start in range(0, len(games), batch_size):
                    batch_end = min(batch_start + batch_size, len(games))
                    game_batch = games[batch_start:batch_end]

                    # Process this batch
                    batch_samples = self._process_game_batch(
                        game_batch, batch_start, max_moves_per_game, num_processes
                    )

                    # Add to current chunk
                    current_chunk_samples.extend(batch_samples)
                    total_samples += len(batch_samples)

                    # Save chunk if it's full
                    while len(current_chunk_samples) >= chunk_size:
                        chunk_to_save = current_chunk_samples[:chunk_size]
                        remaining_samples = current_chunk_samples[chunk_size:]

                        chunk_file = self._save_single_chunk(
                            chunk_to_save, current_chunk_idx
                        )
                        chunk_files.append(chunk_file)
                        current_chunk_idx += 1

                        current_chunk_samples = remaining_samples
                        print(
                            f"""  Saved chunk {current_chunk_idx}, total samples: {
                                total_samples:,}"""
                        )

                    # Clear batch from memory
                    del batch_samples
                    del game_batch

            # Save remaining samples as final chunk
            if current_chunk_samples:
                chunk_file = self._save_single_chunk(
                    current_chunk_samples, current_chunk_idx
                )
                chunk_files.append(chunk_file)
                print(f"  Saved final chunk {current_chunk_idx + 1}")

            # Save chunk metadata
            metadata_file = self._save_chunk_metadata(
                chunk_files, total_samples, chunk_size
            )

            print(f"\nChunked preprocessing complete!")
            print(f"Total samples: {total_samples:,}")
            print(f"Number of chunks: {len(chunk_files)}")

            return chunk_files, metadata_file

        except Exception as e:
            print(f"Error during chunked processing: {e}")
            # Clean up partial chunks
            for chunk_file in chunk_files:
                try:
                    os.remove(chunk_file)
                except:
                    pass
            raise

    def _save_single_chunk(self, chunk_samples, chunk_idx):
        """Save a single chunk to HDF5 file"""
        if not chunk_samples:
            return None

        filename = f"processed_chunk_{chunk_idx:04d}.h5"
        filepath = os.path.join(self.output_dir, filename)

        with h5py.File(filepath, "w") as f:
            board_shape = chunk_samples[0]["board"].shape
            move_shape = chunk_samples[0]["move"].shape

            # Create datasets
            boards = f.create_dataset(
                "boards",
                shape=(len(chunk_samples),) + board_shape,
                dtype=np.float32,
                compression="gzip",
                compression_opts=6,
            )

            moves = f.create_dataset(
                "moves",
                shape=(len(chunk_samples),) + move_shape,
                dtype=np.float32,
                compression="gzip",
                compression_opts=6,
            )

            game_indices = f.create_dataset(
                "game_indices", shape=(len(chunk_samples),), dtype=np.int32
            )

            move_indices = f.create_dataset(
                "move_indices", shape=(len(chunk_samples),), dtype=np.int32
            )

            # Fill datasets
            for i, sample in enumerate(chunk_samples):
                boards[i] = sample["board"]
                moves[i] = sample["move"]
                game_indices[i] = sample["game_idx"]
                move_indices[i] = sample["move_idx"]

            # Save metadata
            f.attrs["num_samples"] = len(chunk_samples)
            f.attrs["board_shape"] = board_shape
            f.attrs["move_shape"] = move_shape
            f.attrs["chunk_idx"] = chunk_idx
            f.attrs["encoder_name"] = self.encoder.__class__.__name__

        return filepath

    def _save_chunk_metadata(self, chunk_files, total_samples, chunk_size):
        """Save metadata for chunked format"""
        metadata_file = os.path.join(self.output_dir, "chunk_metadata.pkl")
        with open(metadata_file, "wb") as f:
            pickle.dump(
                {
                    "chunk_files": chunk_files,
                    "total_samples": total_samples,
                    "chunk_size": chunk_size,
                    "num_chunks": len(chunk_files),
                },
                f,
            )
        return metadata_file

    def resume_from_intermediate(self, save_format="single", chunk_size=10000):
        """Resume preprocessing from intermediate results if available"""
        intermediate_files = [
            f
            for f in os.listdir(self.output_dir)
            if f.startswith("intermediate_results_") and f.endswith(".pkl")
        ]

        if not intermediate_files:
            print("No intermediate files found. Starting fresh preprocessing.")
            return None

        # Find the most recent intermediate file
        latest_file = max(
            intermediate_files,
            key=lambda f: int(f.split("_")[2]) if f.split("_")[
                2].isdigit() else 0,
        )

        print(f"Found intermediate file: {latest_file}")

        try:
            filepath = os.path.join(self.output_dir, latest_file)
            with open(filepath, "rb") as f:
                samples = pickle.load(f)

            print(f"Loaded {len(samples):,} samples from intermediate file")

            # Save in specified format
            if save_format == "single":
                result = self.save_processed_data_hdf5(samples)
            elif save_format == "chunks":
                result = self.save_processed_data_chunks(samples, chunk_size)
            else:
                raise ValueError("save_format must be 'single' or 'chunks'")

            # Clean up intermediate files after successful save
            self._cleanup_intermediate_files()

            return result

        except Exception as e:
            print(f"Error loading intermediate file: {e}")
            return None


def process_single_game_worker(game_data):
    """Worker function for multiprocessing"""
    game_idx, sgf_content, max_moves_per_game, encoder = game_data

    try:
        board_size = 19
        game = GameState.new_game(board_size)

        moves, handicap_info = extract_moves(sgf_content)

        if handicap_info["is_handicap_game"]:
            game, handicap_moves = setup_handicap_game(
                board_size, handicap_info)

        num_moves = len(moves)
        if max_moves_per_game:
            num_moves = min(num_moves, max_moves_per_game)

        game_samples = []

        for move_idx in range(num_moves):
            # Get current board state
            board_tensor = encoder.encode(game)

            # Get the move to be made
            color, (row, col) = moves[move_idx]
            point = Point(row, col)

            # Encode the move
            move_index = encoder.encode_point(point)
            label = np.zeros(encoder.num_points())
            label[move_index] = 1

            game_samples.append(
                {
                    "board": board_tensor.astype(np.float32),
                    "move": label.astype(np.float32),
                    "game_idx": game_idx,
                    "move_idx": move_idx,
                }
            )

            # Apply the move for next iteration
            move_obj = Move.play(point)
            game = game.apply_move(move_obj)

        return game_samples

    except Exception as e:
        print(f"Error processing game {game_idx}: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    from dlgo.encoders.base import get_encoder_by_name

    encoder = get_encoder_by_name("oneplane", 19)
    preprocessor = GoGamePreprocessor(encoder)

    # Preprocess and save all games with memory-efficient processing
    result = preprocessor.preprocess_and_save(
        max_moves_per_game=200,  # Limit moves per game if needed
        save_format="single",  # or "chunks" for large datasets
        num_processes=4,  # Adjust based on your CPU
        batch_size=1000,  # Games to process in each batch
        save_intermediate=True,  # Save progress to prevent data loss
    )

    print(f"Preprocessing complete! Data saved to: {result}")
