import os
import torch

from dlgo.data.data_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.networks import small
from dlgo.networks.trainer import GoTrainer, GoEvaluator


def main():
    board_size = 19
    data_dir = "dlgo/data/go_games/"
    train_games = 1000  # Can now handle much larger datasets
    val_games = 200

    config = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_dir": "checkpoints",
        "num_workers": 4,  # Increase for faster data loading
        "max_moves_per_game": 200,  # Limit moves per game to prevent very long games
    }

    print(f"Using device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Number of workers: {config['num_workers']}")

    encoder = OnePlaneEncoder(board_size)
    print(f"Encoder: {encoder.name()}")
    print(f"Input shape: {encoder.num_planes} x {board_size} x {board_size}")
    print(f"Output size: {encoder.num_points()}")

    processor = GoDataProcessor(
        encoder=encoder.name(), data_directory=data_dir)

    print("\nCreating data loaders...")
    train_loader, val_loader = processor.create_train_val_loaders(
        train_samples=train_games,
        val_samples=val_games,
        batch_size=config["batch_size"],
        shuffle_train=True,
        num_workers=config["num_workers"],
        max_moves_per_game=config["max_moves_per_game"],
        random_seed=42,
    )

    print(f"Train loader: {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader)} batches")

    print("\nTesting data loading...")
    try:
        sample_batch = next(iter(train_loader))
        features, labels = sample_batch
        print(
            f"""Sample batch - Features shape: {features.shape}, Labels shape: {
                labels.shape
            }"""
        )
        print(f"""Features dtype: {
              features.dtype}, Labels dtype: {labels.dtype}""")
        print(
            f"""Memory usage per batch: ~{
                features.numel() * 4 / 1024**2:.1f} MB (features)"""
        )
    except Exception as e:
        print(f"Error loading sample batch: {e}")
        return

    input_shape = (encoder.num_planes, board_size, board_size)
    model = small.SmallNetwork(input_shape)
    print(f"\nModel initialized with input shape: {input_shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    trainer = GoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        save_dir=config["save_dir"],
    )

    os.makedirs(config["save_dir"], exist_ok=True)

    print(f"\nStarting training for {config['num_epochs']} epochs...")
    try:
        trainer.train(num_epochs=config["num_epochs"])
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        return

    print("\nEvaluating model...")
    evaluator = GoEvaluator(model, val_loader, config["device"])

    checkpoint_path = os.path.join(config["save_dir"], "best_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        results = evaluator.evaluate(checkpoint_path=checkpoint_path)
        print(f"\nFinal evaluation results: {results}")
    else:
        print("No checkpoint found for evaluation")


if __name__ == "__main__":
    main()
