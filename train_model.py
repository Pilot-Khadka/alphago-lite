import torch
from torch.nn.parallel import DataParallel

from dlgo.data.fast_data_processor import FastGoDataProcessor
from dlgo.data.data_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.networks import small
from dlgo.networks.trainer import GoTrainer

BATCH_SIZE = 64
NUM_WORKERS = 4


def train_single_gpu(config):
    print("=== Single GPU/CPU Training ===")

    board_size = 19
    encoder = OnePlaneEncoder(board_size)

    processor = GoDataProcessor(
        encoder=encoder.name(), data_directory=config["data_dir"]
    )

    # data loaders (non-distributed)
    processor = FastGoDataProcessor()
    processor.load_preprocessed_data()
    train_loader, val_loader = processor.create_train_val_loaders(
        train_ratio=0.9, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, random_seed=42
    )

    input_shape = (encoder.num_planes, board_size, board_size)
    model = small.SmallNetwork(input_shape)

    if torch.cuda.device_count() > 1 and config["use_data_parallel"]:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

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


def simple_single_gpu_example():
    print("=== Simple Single GPU Example ===")

    config = {
        "data_dir": "dlgo/data/go_games/",
        "train_games": 100,
        "val_games": 20,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_epochs": 10,
        "num_workers": 2,
        "max_moves_per_game": 100,
        "save_dir": "checkpoints_simple",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "use_data_parallel": False,
    }

    model, trainer = train_single_gpu(config)
    return model, trainer


if __name__ == "__main__":
    simple_single_gpu_example()
