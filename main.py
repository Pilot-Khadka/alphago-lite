import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from alphago.train import GoTrainer

# from alphago.networks import SmallNetwork
from alphago.networks import SmallResdualNetwork
from alphago.encoders import OnePlaneEncoder
from alphago.data.data_loader import GoDataset
from alphago.util import load_config


def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def train_ddp(config):
    rank, world_size, local_rank = setup_ddp()

    board_size = 19
    encoder = OnePlaneEncoder(board_size)

    train_dataset = GoDataset(config.train_path)
    val_dataset = GoDataset(config.val_path)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        sampler=val_sampler,
    )

    input_shape = (encoder.num_planes, board_size, board_size)
    # model = SmallNetwork(input_shape).cuda(local_rank)
    model = SmallResdualNetwork(input_shape=input_shape).cuda(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    trainer = GoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device(f"cuda:{local_rank}"),
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        rank=rank,
        world_size=world_size,
        use_ddp=True,
    )

    trainer.train(num_epochs=config.num_epochs)
    dist.destroy_process_group()


def train_single_gpu(config):
    board_size = 19
    encoder = OnePlaneEncoder(board_size)

    train_dataset = GoDataset(config.train_path)
    val_dataset = GoDataset(config.val_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
    )

    input_shape = (encoder.num_planes, board_size, board_size)
    # model = SmallNetwork(input_shape)
    model = SmallResdualNetwork(input_shape=input_shape)

    if torch.cuda.device_count() > 1 and config.use_data_parallel:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = GoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        rank=0,
        world_size=1,
        use_ddp=False,
    )

    trainer.train(num_epochs=config.num_epochs)


def main():
    config = load_config("config/one_plane.yaml")
    config.train_path = "dataset/train"
    config.val_path = "dataset/val"

    if torch.cuda.device_count() > 1:
        print(f"Launching DDP on {torch.cuda.device_count()} GPUs")
        train_ddp(config)
    else:
        print("Using single GPU")
        train_single_gpu(config)


if __name__ == "__main__":
    main()
