import os
import zipfile

# pyrefly: ignore [missing-import]
from src.data.sharding import split_shards
# pyrefly: ignore [missing-import]
from src.data.preprocessor import GoGamePreprocessor
# pyrefly: ignore [missing-import]
from src.encoders.oneplane import OnePlaneEncoder


def main():
    data_dir = "external/computer-go-dataset/Professional/"
    print(os.listdir(data_dir))

    for file in os.listdir(data_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(data_dir, file)
            print(f"Unzipping: {zip_path}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)

    output_dir = "dataset/"

    encoder = OnePlaneEncoder(board_size=13)
    preprocessor = GoGamePreprocessor(
        encoder=encoder,
        data_directory=data_dir,
        output_directory=output_dir,
    )

    # all_games = preprocessor.collect_all_games()
    # print("all games shape:", len(all_games))
    # preprocessor.preprocess(
    #     all_games,
    #     max_moves_per_game=None,
    #     num_processes=os.cpu_count() - 2,
    #     shard_size=500_000,
    # )

    # divide the dataset into train/val/test split
    # collect all dataset first
    # go to the dataset folder
    # collect all .npy shards
    # find total game number
    # split the data into train/val/test
    # place them inside train/val/test folder
    split_shards(output_dir, ratios=(0.8, 0.1, 0.1))


if __name__ == "__main__":
    main()
