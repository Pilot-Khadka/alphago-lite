import os
import torch
from dlgo.data.data_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.networks import small
from dlgo.networks.trainer import GoTrainer, GoEvaluator


def main():
    board_size = 19
    data_dir = "dlgo/data/go_games/"
    train_games = 20000
    val_games = 1000

    encoder = OnePlaneEncoder(board_size)

    processor = GoDataProcessor(
        encoder=encoder.name(), data_directory=data_dir)

    train_loader, val_loader = processor.create_train_val_loaders(
        train_samples=train_games, val_samples=val_games
    )

    input_shape = (encoder.num_planes, board_size, board_size)
    model = small.SmallNetwork(input_shape)

    config = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "checkpoints",
    }
    trainer = GoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        save_dir=config["save_dir"],
    )
    trainer.train(num_epochs=config["num_epochs"])
    evaluator = GoEvaluator(model, val_loader, config["device"])
    results = evaluator.evaluate(
        checkpoint_path=os.path.join(config["save_dir"], "best_checkpoint.pth")
    )
    print(f"\nFinal evaluation results: {results}")


if __name__ == "__main__":
    main()
