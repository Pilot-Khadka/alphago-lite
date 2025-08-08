import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        init_method="env://",
        device_id=device,
    )

    print(f"Rank {rank} initialized on {torch.cuda.current_device()}")


def cleanup_ddp():
    dist.destroy_process_group()


def create_distributed_dataloaders(processor, config, rank, world_size):
    train_dataset = processor.create_dataset(
        data_type="train", max_moves_per_game=config.get("max_moves_per_game", None)
    )

    val_dataset = processor.create_dataset(
        data_type="val", max_moves_per_game=config.get("max_moves_per_game", None)
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
