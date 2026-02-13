from pathlib import Path

import torch
from torch.nn.parallel import DataParallel

from src.train import GoTrainer
from src.networks import SmallNetwork
from src.encoders import OnePlaneEncoder
from src.data.data_loader import GoDataset
from src.util import load_config

BATCH_SIZE = 64
NUM_WORKERS = 4


def train_single_gpu(config):
    board_size = 19
    encoder = OnePlaneEncoder(board_size)
    train_dataset = GoDataset(data_dir="dataset/train")
    val_dataset = GoDataset(data_dir="dataset/val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    input_shape = (encoder.num_planes, board_size, board_size)
    model = SmallNetwork(input_shape)

    if torch.cuda.device_count() > 1 and config["use_data_parallel"]:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = GoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        save_dir=config["save_dir"],
        rank=0,
        world_size=1,
        use_ddp=False,
    )

    trainer.train(num_epochs=config["num_epochs"])

    return model, trainer


def simple_single_gpu_example():
    config = load_config(config_path=Path("config/one_plane.yaml"))

    model, trainer = train_single_gpu(config)
    return model, trainer


if __name__ == "__main__":
    simple_single_gpu_example()
