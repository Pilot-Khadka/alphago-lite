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
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


class GoTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device: torch.device,
        learning_rate=0.001,
        weight_decay: float = 1e-4,
        save_dir: Path = Path("checkpoint"),
        rank=0,
        world_size=1,
        use_ddp=False,
    ):
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.device = device
        self.save_dir = save_dir

        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)

        self.model = model.to(device)

        if use_ddp and world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])
            self.model_without_ddp = self.model.module
        else:
            self.model_without_ddp = self.model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.criterion = nn.CrossEntropyLoss()

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

        # only create tensorboard writer on rank 0
        self.writer = None
        if rank == 0:
            self.writer = SummaryWriter(
                log_dir=f"""runs/go_training_{
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                }"""
            )

        self.train_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

        self.best_val_accuracy = 0.0

    def train_epoch(self, epoch):
        self.model.train()

        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        correct = 0
        total = 0

        if self.rank == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        else:
            pbar = self.train_loader

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            if target.dim() > 1 and target.size(1) > 1:
                target = torch.argmax(target, dim=1)

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            if self.rank == 0 and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100.0 * correct / total:.2f}%",
                    }
                )

            if self.rank == 0 and self.writer and batch_idx % 100 == 0:
                self.writer.add_scalar(
                    "Loss/Train_Batch",
                    loss.item(),
                    epoch * len(self.train_loader) + batch_idx,
                )
        if self.use_ddp and self.world_size > 1:
            # convert to tensors for all_reduce
            loss_tensor = torch.tensor(total_loss, device=self.device)
            correct_tensor = torch.tensor(correct, device=self.device)
            total_tensor = torch.tensor(total, device=self.device)

            # sum across all processes
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

            # avg. loss across processes
            avg_loss = loss_tensor.item() / (len(self.train_loader) * self.world_size)
            accuracy = 100.0 * correct_tensor.item() / total_tensor.item()
        else:
            avg_loss = total_loss / len(self.train_loader)
            accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            if self.rank == 0:
                pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]")
            else:
                pbar = self.val_loader

            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)

                if target.dim() > 1 and target.size(1) > 1:
                    target = torch.argmax(target, dim=1)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                if self.rank == 0 and hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(
                        {
                            "Loss": f"{loss.item():.4f}",
                            "Acc": f"{100.0 * correct / total:.2f}%",
                        }
                    )

        if self.use_ddp and self.world_size > 1:
            loss_tensor = torch.tensor(total_loss, device=self.device)
            correct_tensor = torch.tensor(correct, device=self.device)
            total_tensor = torch.tensor(total, device=self.device)

            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

            avg_loss = loss_tensor.item() / (len(self.val_loader) * self.world_size)
            accuracy = 100.0 * correct_tensor.item() / total_tensor.item()
        else:
            avg_loss = total_loss / len(self.val_loader)
            accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def save_checkpoint(self, epoch, val_accuracy, is_best=False):
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

        checkpoint_path = os.path.join(self.save_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best_checkpoint.pth")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model_without_ddp.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_history = checkpoint["train_history"]
        self.best_val_accuracy = checkpoint["val_accuracy"]

        return checkpoint["epoch"]

    def train(self, num_epochs, resume_from=None):
        start_epoch = 0

        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            if self.rank == 0:
                print(f"Resuming training from epoch {start_epoch}")

        if self.rank == 0:
            print(f"Starting training for {num_epochs} epochs")
            print(f"Device: {self.device}")
            print(f"World size: {self.world_size}")
            print(
                f"""Model parameters: {
                    sum(p.numel() for p in self.model_without_ddp.parameters()):,}"""
            )

        start_time = time.time()

        for epoch in range(start_epoch, num_epochs):
            if self.rank == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print("-" * 50)

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

                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                print(f"Learning Rate: {current_lr:.6f}")

                is_best = val_acc > self.best_val_accuracy
                if is_best:
                    self.best_val_accuracy = val_acc

                self.save_checkpoint(epoch, val_acc, is_best)

                # if self.should_early_stop(epoch):
                #     print("Early stopping triggered")
                #     break

        if self.rank == 0:
            total_time = time.time() - start_time
            print(f"\nTraining completed in {total_time:.2f} seconds")
            print(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")

            self.save_training_history()
            if self.writer:
                self.writer.close()

    def should_early_stop(self, epoch, patience=10):
        if epoch < patience:
            return False

        recent_val_acc = self.train_history["val_accuracy"][-patience:]
        return max(recent_val_acc) <= self.best_val_accuracy

    def save_training_history(self):
        if self.rank != 0:
            return

        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.train_history, f, indent=2)
