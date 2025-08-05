import os
import torch

import torch.multiprocessing as mp
from train_model import train_single_gpu
from dlgo.data.data_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.networks.ddp import setup_ddp, cleanup_ddp
from dlgo.networks.trainer import GoTrainer, GoEvaluator
from dlgo.networks import small


def train_ddp_worker(rank, world_size, config):
    """Worker function for DDP training"""
    print(f"Starting DDP worker {rank}/{world_size}")

    setup_ddp(rank, world_size)

    try:
        board_size = 19
        encoder = OnePlaneEncoder(board_size)

        processor = GoDataProcessor(
            encoder=encoder.name(), data_directory=config["data_dir"]
        )

        # train/val split (only on rank 0 to avoid race conditions)
        if rank == 0:
            processor._create_train_val_split(
                train_samples=config["train_games"],
                val_samples=config["val_games"],
                random_seed=42,
            )

        # synchronize all processes
        torch.distributed.barrier()

        if rank != 0:
            processor._load_all_games()
            # recteate the same split (deterministic due to same random seed)
            processor._create_train_val_split(
                train_samples=config["train_games"],
                val_samples=config["val_games"],
                random_seed=42,
            )

        from torch.utils.data.distributed import DistributedSampler

        from dlgo.data.data_processor import GoGameDataset

        train_dataset = GoGameDataset(
            processor.train_games,
            processor.encoder,
            config.get("max_moves_per_game", None),
        )
        val_dataset = GoGameDataset(
            processor.val_games,
            processor.encoder,
            config.get("max_moves_per_game", None),
        )

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"]
            // world_size,  # Divide batch size across GPUs
            sampler=train_sampler,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config["batch_size"] // world_size,
            sampler=val_sampler,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=False,
        )

        input_shape = (encoder.num_planes, board_size, board_size)
        model = small.SmallNetwork(input_shape)

        trainer = GoTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=f"cuda:{rank}",
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            save_dir=config["save_dir"],
            rank=rank,
            world_size=world_size,
            use_ddp=True,
        )

        trainer.train(num_epochs=config["num_epochs"])

    finally:
        cleanup_ddp()


def train_multi_gpu_ddp(config):
    print("=== Multi-GPU DDP Training ===")

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for DDP training")
    mp.spawn(train_ddp_worker, args=(world_size, config),
             nprocs=world_size, join=True)


def main():
    config = {
        "data_dir": "dlgo/data/go_games/",
        "train_games": None,
        "val_games": None,
        #  batch size (will be divided across GPUs in DDP)
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "num_workers": 4,
        "max_moves_per_game": 200,
        "save_dir": "checkpoints",
        "use_data_parallel": False,
        "use_ddp": True,  # set to False for single GPU training
    }

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config["device"] = "cpu"
        config["use_ddp"] = False
        config["use_data_parallel"] = False
    else:
        config["device"] = "cuda"
        print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")

    os.makedirs(config["save_dir"], exist_ok=True)

    if config["use_ddp"] and torch.cuda.device_count() > 1:
        train_multi_gpu_ddp(config)
    else:
        model, trainer = train_single_gpu(config)

        print("\nEvaluating trained model...")
        val_loader = trainer.val_loader
        evaluator = GoEvaluator(model, val_loader, config["device"])

        checkpoint_path = os.path.join(
            config["save_dir"], "best_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            results = evaluator.evaluate(checkpoint_path=checkpoint_path)
            print(f"Final evaluation results: {results}")


if __name__ == "__main__":
    main()
