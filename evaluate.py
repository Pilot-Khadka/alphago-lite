from alphago.train import GoTrainer
from alphago.networks import SmallResidualNetwork
from alphago.encoders import OnePlaneEncoder
from alphago.data.data_loader import GoDataset
from alphago.util import load_config


def main():
    config = load_config("config/one_plane.yaml")

    print("[INFO] Config loaded:")
    config.dump()

    encoder = OnePlaneEncoder(config.data.board_size)
    model = SmallResidualNetwork(
        channel_size=encoder.num_planes, board_size=config.data.board_size
    )

    trainer = GoTrainer(
        model=model,
        val_dataset=GoDataset(config.data.val_path),
        batch_size=config.loader.batch_size,
        learning_rate=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    trainer.load_checkpoint("checkpoint/best.pth")
    trainer.validate()


if __name__ == "__main__":
    main()
