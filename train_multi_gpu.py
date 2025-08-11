import os
import argparse
import torch
import torch.multiprocessing as mp

from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.data.ddp import setup_ddp, cleanup_ddp, create_distributed_dataset_and_loaders
from dlgo.networks.trainer import GoTrainer, GoEvaluator
from dlgo.networks import small
from dlgo.data.data_loader import GoDataProcessor


def train_ddp_worker(rank, world_size, config):
    print(f"Starting DDP worker {rank}/{world_size}")
    setup_ddp(rank, world_size)

    try:
        board_size = 19
        encoder = OnePlaneEncoder(board_size)

        processor = GoDataProcessor(data_directory=config["processed_data_dir"])

        if rank == 0:
            print("Creating train/val split...")
            split_info = processor.create_train_val_test_split(
                train_ratio=config["train_ratio"],
                val_ratio=config["val_ratio"],
                test_ratio=config.get("test_ratio", 0.0),
                random_seed=config.get("random_seed", 42),
            )
            print(f"Split info: {split_info}")

        torch.distributed.barrier()

        if rank != 0:
            processor.create_train_val_test_split(
                train_ratio=config["train_ratio"],
                val_ratio=config["val_ratio"],
                test_ratio=config.get("test_ratio", 0.0),
                random_seed=config.get("random_seed", 42),
            )

        train_loader, val_loader, train_sampler, val_sampler = (
            create_distributed_dataset_and_loaders(processor, config, rank, world_size)
        )

        if rank == 0:
            print(f"Train loader: {len(train_loader)} batches per GPU")
            print(f"Val loader: {len(val_loader)} batches per GPU")

            stats = processor.get_data_stats()
            print(f"Data statistics: {stats}")

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


def train_multi_gpu_ddp_npz(config):
    print("=== Multi-GPU DDP Training with NPZ Data Loading ===")

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for DDP training")

    processor = GoDataProcessor(data_directory=config["processed_data_dir"])

    try:
        stats = processor.get_data_stats()
        print("NPZ data found and validated")
        print(f"Data info: {stats}")
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure NPZ processed data is available before training.")
        return None

    mp.spawn(train_ddp_worker, args=(world_size, config), nprocs=world_size, join=True)


def train_single_gpu_npz(config):
    print("=== Single GPU Training with NPZ Data Loading ===")

    processor = GoDataProcessor(data_directory=config["processed_data_dir"])

    stats = processor.get_data_stats()
    print(f"Data info: {stats}")

    train_loader, val_loader = processor.create_train_val_loaders(
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config.get("test_ratio", 0.0),
        batch_size=config["batch_size"],
        shuffle_train=True,
        num_workers=config["num_workers"],
        random_seed=config.get("random_seed", 42),
        max_samples_per_game=config.get("max_samples_per_game", None),
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")

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


def evaluate_model_npz(config):
    print("Evaluating trained model...")

    processor = GoDataProcessor(data_directory=config["processed_data_dir"])

    processor.create_train_val_test_split(
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config.get("test_ratio", 0.0),
        random_seed=config.get("random_seed", 42),
    )

    val_loader = processor.create_dataloader(
        data_type="val",
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        max_samples_per_game=config.get("max_samples_per_game", None),
    )

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
    default_config = {
        "processed_data_dir": "processed_go_data",
        "train_ratio": 0.8,
        "val_ratio": 0.2,
        "test_ratio": 0.0,
        "max_samples_per_game": None,
        "random_seed": 42,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_epochs": 5,
        "num_workers": 4,
        "save_dir": "checkpoints",
        "use_ddp": True,
    }

    parser = argparse.ArgumentParser(description="Train a Go game AI model")

    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default=default_config["processed_data_dir"],
        help="Directory containing processed NPZ files",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=default_config["train_ratio"],
        help="Ratio of data to use for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=default_config["val_ratio"],
        help="Ratio of data to use for validation",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=default_config["test_ratio"],
        help="Ratio of data to use for testing",
    )
    parser.add_argument(
        "--max_samples_per_game",
        type=int,
        default=default_config["max_samples_per_game"],
        help="Maximum samples to use per game (None for all)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=default_config["random_seed"],
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_config["batch_size"],
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=default_config["learning_rate"],
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=default_config["weight_decay"],
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=default_config["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=default_config["num_workers"],
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=default_config["save_dir"],
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--no_ddp",
        action="store_false",
        dest="use_ddp",
        help="Disable Distributed Data Parallel (DDP) training",
    )

    args = parser.parse_args()

    config = {
        "processed_data_dir": args.processed_data_dir,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "max_samples_per_game": args.max_samples_per_game,
        "random_seed": args.random_seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "save_dir": args.save_dir,
        "use_ddp": args.use_ddp,
    }

    # Handle device configuration
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config["device"] = "cpu"
        config["use_ddp"] = False
        config["num_workers"] = min(config["num_workers"], 2)  # Reduce workers for CPU
        # Reduce batch size for CPU
        config["batch_size"] = min(config["batch_size"], 64)
    else:
        config["device"] = "cuda"
        print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")

    os.makedirs(config["save_dir"], exist_ok=True)

    try:
        processor = GoDataProcessor(data_directory=config["processed_data_dir"])
        if not processor.verify_data_integrity(num_samples=5):
            print("Data integrity check failed!")
            return
    except Exception as e:
        print(f"Error validating NPZ data: {e}")
        print("Please run preprocessing first to generate NPZ files.")
        return

    if config["use_ddp"] and torch.cuda.device_count() > 1:
        train_multi_gpu_ddp_npz(config)
        evaluate_model_npz(config)
    else:
        model, trainer = train_single_gpu_npz(config)
        evaluate_model_npz(config)


if __name__ == "__main__":
    main()
