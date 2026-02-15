import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from alphago.train import GoTrainer
# from alphago.networks import SmallNetwork

from alphago.networks import SmallResidualNetwork
from alphago.encoders import OnePlaneEncoder
from alphago.data.data_loader import GoDataset
from alphago.util import load_config


def evaluate_single_gpu(config):
    train_dataset = GoDataset(config.data.train_path)
    val_dataset = GoDataset(config.data.val_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.loader.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.loader.batch_size,
        shuffle=False,
        num_workers=4,
    )

    encoder = OnePlaneEncoder(config.data.board_size)

    # model = SmallNetwork(
    #     num_channels=encoder.num_planes, board_size=config.data.board_size
    # )
    model = SmallResidualNetwork(
        channel_size=encoder.num_planes, board_size=config.data.board_size
    )

    if torch.cuda.device_count() > 1 and config.train.use_data_parallel:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = GoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
        rank=0,
        world_size=1,
        use_ddp=False,
    )

    trainer.load_checkpoint(checkpoint_path="checkpoint/best_checkpoint.pth")
    trainer.validate(epoch=10)


def main():
    config = load_config("config/one_plane.yaml")
    config.data.train_path = "dataset/train"
    config.data.val_path = "dataset/val"

    print("[INFO] Config loaded:")
    config.dump()

    print("Using single GPU")
    evaluate_single_gpu(config)


if __name__ == "__main__":
    main()
