from typing import Optional, Union

import os
import time
import json

from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


class GoTrainer:
    train_loader: Optional[DataLoader]
    val_loader: DataLoader
    model: Union[DDP, torch.nn.DataParallel, nn.Module]

    def __init__(
        self,
        model: nn.Module,
        val_dataset: Dataset,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        train_dataset: Dataset | None = None,
        num_workers: int = 4,
        save_dir: Path = Path("checkpoint/dl"),
        is_notebook: bool = False,
    ):
        self.save_dir = save_dir
        self.disable_tqdm = is_notebook
        self.start_epoch = 0

        self._setup_distributed()
        self._setup_data(train_dataset, val_dataset, batch_size, num_workers)
        self._setup_model(model)
        self._setup_optimizer(learning_rate, weight_decay)
        self._setup_logging()

    def _setup_distributed(self):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.use_ddp = local_rank >= 0

        if self.use_ddp:
            dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = local_rank
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_data(self, train_dataset, val_dataset, batch_size, num_workers):
        if train_dataset is not None:
            train_sampler = DistributedSampler(train_dataset) if self.use_ddp else None
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=train_sampler is None,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            self.train_loader = None

        val_sampler = (
            DistributedSampler(val_dataset, shuffle=False) if self.use_ddp else None
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def _setup_model(self, model: nn.Module):
        if self.use_ddp:
            model = model.cuda(self.local_rank)
            self.model = DDP(
                model, device_ids=[self.local_rank], output_device=self.local_rank
            )
        else:
            model = model.to(self.device)
            if torch.cuda.device_count() > 1:
                print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
                self.model = torch.nn.DataParallel(model)
            else:
                self.model = model

        self.model_without_ddp = (
            self.model.module
            if isinstance(self.model, (DDP, torch.nn.DataParallel))
            else self.model
        )

    def _setup_optimizer(self, learning_rate: float, weight_decay: float):
        self.optimizer = optim.Adam(
            self.model_without_ddp.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )
        self.best_val_accuracy = 0.0

    def _setup_logging(self):
        self.train_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        self.writer = None
        if self.rank == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=f"runs/go_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    def _reduce_metrics(
        self, total_loss: float, correct: int, total: int, num_batches: int
    ):
        if self.use_ddp and self.world_size > 1:
            t = torch.tensor(
                [total_loss, float(correct), float(total)], device=self.device
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss, correct, total = t.tolist()
            avg_loss = total_loss / (num_batches * self.world_size)
        else:
            avg_loss = total_loss / num_batches

        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def _prepare_targets(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() > 1 and target.size(1) > 1:
            return torch.argmax(target, dim=1)
        return target

    def train_epoch(self, epoch: int):
        self.model.train()
        assert self.train_loader is not None

        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1} [Train]",
            disable=self.rank != 0 or self.disable_tqdm,
        )

        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.device)
            target = self._prepare_targets(target.to(self.device))

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            predicted = output.argmax(dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if self.rank == 0:
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * correct / total:.2f}%",
                    }
                )
                if self.writer and batch_idx % 100 == 0:
                    self.writer.add_scalar(
                        "Loss/Train_Batch",
                        loss.item(),
                        epoch * len(self.train_loader) + batch_idx,
                    )

        avg_loss, accuracy = self._reduce_metrics(
            total_loss, correct, total, len(self.train_loader)
        )

        if self.rank == 0:
            print(f"Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%")

        return avg_loss, accuracy

    def validate(self, epoch: int = 0):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1} [Val]",
            disable=self.rank != 0 or self.disable_tqdm,
        )

        with torch.no_grad():
            for data, target in pbar:
                data = data.to(self.device)
                target = self._prepare_targets(target.to(self.device))

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                predicted = output.argmax(dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                if self.rank == 0:
                    pbar.set_postfix(
                        {
                            "Loss": f"{loss.item():.4f}",
                            "Acc": f"{100.0 * correct / total:.2f}%",
                        }
                    )

        avg_loss, accuracy = self._reduce_metrics(
            total_loss, correct, total, len(self.val_loader)
        )

        if self.rank == 0:
            print(f"Val Loss: {avg_loss:.4f}, Val Acc: {accuracy:.2f}%")

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, val_accuracy: float, is_best: bool = False):
        if self.rank != 0:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model_without_ddp.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_accuracy": val_accuracy,
            "train_history": self.train_history,
        }

        torch.save(checkpoint, os.path.join(self.save_dir, "last.pth"))

        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, "best.pth"))
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = self._normalize_state_dict(
            checkpoint.get("model_state_dict", checkpoint), ddp_wrapped=self.use_ddp
        )

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_history = checkpoint["train_history"]
        self.best_val_accuracy = checkpoint["val_accuracy"]
        self.start_epoch = checkpoint["epoch"] + 1

    @staticmethod
    def _normalize_state_dict(state_dict: dict, ddp_wrapped: bool) -> dict:
        """
        Checkpoints can come from DDP or non-DDP runs, so keys may or may not have
        "module." prefixes. This strips all existing prefixes and re-adds one only
        when loading into a DDP-wrapped model.
        """
        normalized = {}
        for k, v in state_dict.items():
            while k.startswith("module."):
                k = k[len("module.") :]
            if ddp_wrapped:
                k = "module." + k
            normalized[k] = v
        return normalized

    def train(self, num_epochs: int):
        if self.train_loader is None:
            raise RuntimeError("train_dataset was not provided, cannot call train()")

        if self.rank == 0:
            print(f"Starting training for {num_epochs} epochs")
            print(f"Device: {self.device}, World size: {self.world_size}")
            print(
                f"Model parameters: {sum(p.numel() for p in self.model_without_ddp.parameters()):,}"
            )

        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, num_epochs):
                if self.rank == 0:
                    print(f"\nEpoch {epoch + 1}/{num_epochs}\n" + "-" * 50)

                train_loss, train_acc = self.train_epoch(epoch)
                val_loss, val_acc = self.validate(epoch)
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

                if self.rank == 0:
                    self.train_history["loss"].append(train_loss)
                    self.train_history["accuracy"].append(train_acc)
                    self.train_history["val_loss"].append(val_loss)
                    self.train_history["val_accuracy"].append(val_acc)
                    self.train_history["learning_rate"].append(current_lr)

                    if self.writer:
                        self.writer.add_scalar("Loss/Train", train_loss, epoch)
                        self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                        self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
                        self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)
                        self.writer.add_scalar("Learning_Rate", current_lr, epoch)

                    print(f"Learning Rate: {current_lr:.6f}")

                    is_best = val_acc > self.best_val_accuracy
                    if is_best:
                        self.best_val_accuracy = val_acc
                    self.save_checkpoint(epoch, val_acc, is_best)

        finally:
            if self.use_ddp:
                dist.destroy_process_group()

            if self.rank == 0:
                total_time = time.time() - start_time
                print(f"\nTraining completed in {total_time:.2f}s")
                print(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
                self._save_training_history()
                if self.writer:
                    self.writer.close()

    def _save_training_history(self):
        if self.rank != 0:
            return
        with open(os.path.join(self.save_dir, "training_history.json"), "w") as f:
            json.dump(self.train_history, f, indent=2)
