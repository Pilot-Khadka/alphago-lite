from dlgo.data.data_processor import GoDataProcessor

if __name__ == "__main__":
    data_dir = "dlgo/data/go_games/"
    processor = GoDataProcessor(encoder="oneplane", data_directory=data_dir)

    train_loader, val_loader = processor.create_train_val_loaders(
        train_samples=100, val_samples=100
    )
    print("train loader:", train_loader)

    train_iter = iter(train_loader)
    x, y = next(train_iter)
    print("x:", x)
    print("y:", y)
