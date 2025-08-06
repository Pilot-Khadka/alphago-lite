import os
import torch

import torch.multiprocessing as mp
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.networks.ddp import setup_ddp, cleanup_ddp
from dlgo.networks.trainer import GoTrainer, GoEvaluator
from dlgo.networks import small
from dlgo.data.fast_data_processor import FastGoDataProcessor
from torch.utils.data.distributed import DistributedSampler


def train_ddp_worker(rank, world_size, config):
    print(f"Starting DDP worker {rank}/{world_size}")
    setup_ddp(rank, world_size)

    try:
        board_size = 19
        encoder = OnePlaneEncoder(board_size)

        processor = FastGoDataProcessor(
            processed_data_directory=config["processed_data_dir"]
        )

        # load data only on rank 0 to avoid race condition
        if rank == 0:
            print("Loading preprocessed data...")
            processor.load_preprocessed_data(
                data_file=config.get("data_file", None),
                metadata_file=config.get("metadata_file", None),
            )

            processor.create_train_val_split(
                train_ratio=config["train_ratio"],
                random_seed=42,
            )

        torch.distributed.barrier()

        # load data on other ranks
        if rank != 0:
            processor.load_preprocessed_data(
                data_file=config.get("data_file", None),
                metadata_file=config.get("metadata_file", None),
            )

            processor.create_train_val_split(
                train_ratio=config["train_ratio"],
                random_seed=42,
            )

        train_sampler = DistributedSampler(
            torch.utils.data.Subset(processor.dataset, processor.train_indices),
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

        val_sampler = DistributedSampler(
            torch.utils.data.Subset(processor.dataset, processor.val_indices),
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(processor.dataset, processor.train_indices),
            batch_size=config["batch_size"] // world_size,
            sampler=train_sampler,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if config["num_workers"] > 0 else False,
        )

        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(processor.dataset, processor.val_indices),
            batch_size=config["batch_size"] // world_size,
            sampler=val_sampler,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if config["num_workers"] > 0 else False,
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

        if rank == 0:
            print("\nData loading info:")
            print(processor.get_data_info())

    finally:
        cleanup_ddp()


def train_multi_gpu_ddp_fast(config):
    print("=== Multi-GPU DDP Training with Fast Data Loading ===")

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for DDP training")

    processor = FastGoDataProcessor(
        processed_data_directory=config["processed_data_dir"]
    )

    try:
        processor.load_preprocessed_data(
            data_file=config.get("data_file", None),
            metadata_file=config.get("metadata_file", None),
        )
        print("Preprocessed data found and validated")
        print("Data info:", processor.get_data_info())
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure preprocessed data is available before training.")
        return None, None

    mp.spawn(train_ddp_worker, args=(world_size, config), nprocs=world_size, join=True)


def train_single_gpu_fast(config):
    print("=== Single GPU Training with Fast Data Loading ===")

    processor = FastGoDataProcessor(
        processed_data_directory=config["processed_data_dir"]
    )

    print("Loading preprocessed data...")
    processor.load_preprocessed_data(
        data_file=config.get("data_file", None),
        metadata_file=config.get("metadata_file", None),
    )

    print("Data info:", processor.get_data_info())

    train_loader, val_loader = processor.create_train_val_loaders(
        train_ratio=config["train_ratio"],
        batch_size=config["batch_size"],
        shuffle_train=True,
        num_workers=config["num_workers"],
        random_seed=42,
    )

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


def main():
    config = {
        "processed_data_dir": "processed_go_data",
        "data_file": None,
        "metadata_file": None,
        "train_ratio": 0.9,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "num_workers": 4,
        "save_dir": "checkpoints",
        "use_ddp": True,
    }

    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config["device"] = "cpu"
        config["use_ddp"] = False
        config["num_workers"] = 2
    else:
        config["device"] = "cuda"
        print(f"CUDA available with {torch.cuda.device_count()} GPU(s)")

    os.makedirs(config["save_dir"], exist_ok=True)

    if config["use_ddp"] and torch.cuda.device_count() > 1:
        train_multi_gpu_ddp_fast(config)

        print("\nEvaluating trained model...")

        processor = FastGoDataProcessor(
            processed_data_directory=config["processed_data_dir"]
        )
        processor.load_preprocessed_data(
            data_file=config.get("data_file", None),
            metadata_file=config.get("metadata_file", None),
        )
        processor.create_train_val_split(
            train_ratio=config["train_ratio"],
            random_seed=42,
        )

        val_loader = processor.create_dataloader(
            data_type="val",
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
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
        else:
            print("No checkpoint found for evaluation")
    else:
        model, trainer = train_single_gpu_fast(config)

        print("\nEvaluating trained model...")
        val_loader = trainer.val_loader
        evaluator = GoEvaluator(model, val_loader, config["device"])

        checkpoint_path = os.path.join(config["save_dir"], "best_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            results = evaluator.evaluate(checkpoint_path=checkpoint_path)
            print(f"Final evaluation results: {results}")


if __name__ == "__main__":
    main()
