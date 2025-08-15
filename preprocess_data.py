from dlgo.encoders.base import get_encoder_by_name
from dlgo.data.preprocess_games import GoGamePreprocessor


ENCODER_NAME = "oneplane"
BOARD_SIZE = 19
MAX_MOVES_PER_GAME = 300  # None for no limit
NUM_PROCESSES = 4
BATCH_SIZE = 1000
NUM_WORKERS = 4


def preprocess_data():
    print("=" * 60)
    print("STEP 1: PREPROCESSING RAW GAMES")
    print("=" * 60)

    encoder = get_encoder_by_name(ENCODER_NAME, BOARD_SIZE)

    preprocessor = GoGamePreprocessor(
        encoder=encoder,
        data_directory="go_data/go_games/",
        output_directory="processed_go_data",
        shard_size=200000,  # 200k positions per shard
        use_float16=True,  # Reduce memory usage
        bitpack_boards=False,  # Set to True for binary board encodings
    )

    print("Starting preprocessing...")
    preprocessor.preprocess_to_shards(
        max_moves_per_game=300,  # Limit game length
        num_processes=4,
        batch_size=5000,
    )


def main():
    print("Go Game Fast Data Loading System")
    print("=" * 60)
    print("Running in PREPROCESSING mode...")
    preprocess_data()
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")


if __name__ == "__main__":
    main()
