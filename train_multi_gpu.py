import os
import torch
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from alphago.networks import small
from alphago.data.ddp import setup_ddp, cleanup_ddp
from alphago.encoders.oneplane import OnePlaneEncoder
from alphago.data.data_loader import GoShardDataProcessor
from alphago.networks.trainer import GoTrainer, GoEvaluator


def create_distributed_dataset_and_loaders(processor, config, rank, world_size):
    train_dataset = processor.create_dataset(
        data_type="train",
        max_samples=config.get("max_samples_train", None),
        move_as_index=True,  # for CrossEntropyLoss
    )

    val_dataset = processor.create_dataset(
        data_type="val",
        max_samples=config.get("max_samples_val", None),
        move_as_index=True,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    # Adjust batch size per GPU
    per_gpu_batch_size = config["batch_size"] // world_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=per_gpu_batch_size,
        sampler=val_sampler,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    if rank == 0:
        print(f"Per-GPU batch size: {per_gpu_batch_size}")
        print(f"Train dataset size: {len(train_dataset):,}")
        print(f"Val dataset size: {len(val_dataset):,}")
        print(f"Train batches per GPU: {len(train_loader):,}")
        print(f"Val batches per GPU: {len(val_loader):,}")

    return train_loader, val_loader, train_sampler, val_sampler


def train_ddp_worker(rank, world_size, config):
    setup_ddp(rank, world_size)

    try:
        board_size = 19
        encoder = OnePlaneEncoder(board_size)

        processor = GoShardDataProcessor(data_directory=config["processed_data_dir"])

        # only rank 0 handles split and print info
        if rank == 0:
            print("Creating train/val split...")
            split_info = processor.create_train_val_test_split(
                train_ratio=config["train_ratio"],
                val_ratio=config["val_ratio"],
                test_ratio=config.get("test_ratio", 0.0),
                random_seed=config.get("random_seed", 42),
                max_total_samples=config.get("max_total_samples", None),
            )
            print(f"Split info: {split_info}")
            stats = processor.get_data_stats()
            print(f"Data statistics: {stats}")

        torch.distributed.barrier()

        # all other ranks create the same split
        # deterministic due tot randim seed
        if rank != 0:
            processor.create_train_val_test_split(
                train_ratio=config["train_ratio"],
                val_ratio=config["val_ratio"],
                test_ratio=config.get("test_ratio", 0.0),
                random_seed=config.get("random_seed", 42),
                max_total_samples=config.get("max_total_samples", None),
            )

        train_loader, val_loader, train_sampler, val_sampler = (
            create_distributed_dataset_and_loaders(processor, config, rank, world_size)
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

        if hasattr(trainer, "set_epoch_samplers"):
            trainer.set_epoch_samplers(train_sampler, val_sampler)

        trainer.train(num_epochs=config["num_epochs"])

    except Exception as e:
        print(f"Error in DDP worker {rank}: {e}")
        raise e
    finally:
        cleanup_ddp()


def train_multi_gpu_ddp_shard(config):
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for DDP training")

    processor = GoShardDataProcessor(data_directory=config["processed_data_dir"])

    try:
        stats = processor.get_data_stats()
        print("Shard data found and validated")
        print(f"Data info: {stats}")

        max_total = config.get("max_total_samples", None)
        if max_total:
            print(
                f"""Using subset of {max_total:,} samples out of {
                    stats["total_positions"]:,} total"""
            )
        else:
            print(f"Using all {stats['total_positions']:,} samples")

    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure shard processed data is available before training.")
        return None

    mp.spawn(train_ddp_worker, args=(world_size, config), nprocs=world_size, join=True)


def train_single_gpu_shard(config):
    processor = GoShardDataProcessor(data_directory=config["processed_data_dir"])

    stats = processor.get_data_stats()
    print(f"Data info: {stats}")

    split_info = processor.create_train_val_test_split(
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config.get("test_ratio", 0.0),
        random_seed=config.get("random_seed", 42),
        max_total_samples=config.get("max_total_samples", None),  # Fixed parameter name
    )
    print(f"Split info: {split_info}")

    train_loader, val_loader = processor.create_train_val_loaders(
        # these ratio are ignored since split is already created
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config.get("test_ratio", 0.0),
        batch_size=config["batch_size"],
        shuffle_train=True,
        num_workers=config["num_workers"],
        random_seed=config.get("random_seed", 42),
        max_total_samples=config.get("max_total_samples", None),
        move_as_index=True,  # CrossEntropyLoss
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Total train samples: {len(train_loader.dataset):,}")
    print(f"Total val samples: {len(val_loader.dataset):,}")

    board_size = 19
    encoder = OnePlaneEncoder(board_size)
    input_shape = (encoder.num_planes, board_size, board_size)
    model = small.SmallNetwork(input_shape)

    trainer = GoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        save_dir=config["save_dir"],
        rank=0,
        world_size=1,
        use_ddp=False,
    )

    trainer.train(num_epochs=config["num_epochs"])

    return model, trainer


def evaluate_model_shard(config):
    processor = GoShardDataProcessor(data_directory=config["processed_data_dir"])

    processor.create_train_val_test_split(
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config.get("test_ratio", 0.0),
        random_seed=config.get("random_seed", 42),
        max_total_samples=config.get("max_total_samples", None),
    )

    val_loader = processor.create_dataloader(
        data_type="val",
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        # Fixed parameter name
        max_samples=config.get("max_samples_val", None),
        move_as_index=True,
    )

    print(f"Evaluation on {len(val_loader.dataset):,} validation samples")

    board_size = 19
    encoder = OnePlaneEncoder(board_size)
    input_shape = (encoder.num_planes, board_size, board_size)
    model = small.SmallNetwork(input_shape)

    evaluator = GoEvaluator(model, val_loader, config["device"])

    checkpoint_path = os.path.join(config["save_dir"], "best_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        results = evaluator.evaluate(checkpoint_path=checkpoint_path)
        print(f"Final evaluation results: {results}")
        return results
    else:
        print("No checkpoint found for evaluation")
        return None


def main():
    config = {
        "processed_data_dir": "processed_go_data",
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "test_ratio": 0.0,
        "max_total_samples": None,
        "max_samples_train": None,  # No additional limit on train samples
        "max_samples_val": None,  # No additional limit on val samples
        "random_seed": 42,
        "batch_size": 8192,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_epochs": 5,
        "num_workers": 4,
        "save_dir": "checkpoints",
        "use_ddp": True,
    }

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config["device"] = "cpu"
        config["use_ddp"] = False
        config["num_workers"] = 2
        config["batch_size"] = 64
        config["max_total_samples"] = 10000  # smaller dataset for CPU
    else:
        config["device"] = "cuda"
        print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")

    os.makedirs(config["save_dir"], exist_ok=True)

    try:
        processor = GoShardDataProcessor(data_directory=config["processed_data_dir"])

        if not processor.verify_data_integrity(num_samples=100):
            print("Data integrity check failed!")
            return

        stats = processor.get_data_stats()
        print(f"\nAvailable data: {stats['total_positions']:,} positions")
        if config.get("max_total_samples"):
            usage_percent = (
                config["max_total_samples"] / stats["total_positions"]
            ) * 100
            print(
                f"""Will use {config["max_total_samples"]:,} samples ({
                    usage_percent:.1f}% of available data)"""
            )
        else:
            print("Will use all available data")

    except Exception as e:
        print(f"Error validating shard data: {e}")
        print("Please run shard preprocessing first to generate shard files.")
        return

    if config["use_ddp"] and torch.cuda.device_count() > 1:
        print(f"\nStarting multi-GPU training on {torch.cuda.device_count()} GPUs...")
        train_multi_gpu_ddp_shard(config)
        evaluate_model_shard(config)
    else:
        print("\nStarting single-GPU training...")
        model, trainer = train_single_gpu_shard(config)
        evaluate_model_shard(config)


if __name__ == "__main__":
    main()
