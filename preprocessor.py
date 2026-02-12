from src.data.preprocess_games import GoGamePreprocessor
from src.encoders.oneplane import OnePlaneEncoder

if __name__ == "__main__":
    encoder = OnePlaneEncoder(board_size=13)
    preprocessor = GoGamePreprocessor(
        encoder=encoder,
        data_directory="external/",
        output_directory="dataset/",
    )
    preprocessor._collect_all_games()
