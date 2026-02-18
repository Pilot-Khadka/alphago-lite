import argparse

from alphago.train import GoTrainer
from alphago.networks import SmallResidualNetwork
from alphago.encoders import OnePlaneEncoder
from alphago.data.data_loader import GoDataset
from alphago.util import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--is_notebook", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config("config/one_plane.yaml")
    config.resume = args.resume
    config.checkpoint_path = "checkpoint/best.pth"

    print("[INFO] Config loaded:")
    config.dump()

    encoder = OnePlaneEncoder(config.data.board_size)
    model = SmallResidualNetwork(
        channel_size=encoder.num_planes, board_size=config.data.board_size
    )

    trainer = GoTrainer(
        model=model,
        train_dataset=GoDataset(config.data.train_path),
        val_dataset=GoDataset(config.data.val_path),
        batch_size=config.loader.batch_size,
        learning_rate=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        is_notebook=args.is_notebook,
    )

    if config.resume:
        trainer.load_checkpoint(config.checkpoint_path)

    trainer.train(num_epochs=config.train.num_epochs)


if __name__ == "__main__":
    main()
