import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..runs import Run


def warmup_cosine(warmup_steps: int, total_steps: int, min_factor: float = 0.01):
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


class PolicyTrainer:
    """
    supervised policy training on a stream of (planes, move) pairs
    step based, so a run can stop at a time limit and resume in a new session
    """

    def __init__(
        self,
        run: Run,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        cfg: dict,
    ):
        self.run = run
        # pyrefly: ignore [no-matching-overload]
        self.model = model.to(device).to(memory_format=torch.channels_last)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.amp = bool(cfg.get("amp", True)) and device.type == "cuda"

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg["lr"],
            momentum=cfg.get("momentum", 0.9),
            nesterov=True,
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, warmup_cosine(cfg.get("warmup_steps", 1000), cfg["total_steps"])
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.step = 0
        self.best_val = None

    def state(self) -> dict:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val": self.best_val,
        }

    def load_state(self, checkpoint: dict) -> None:
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_val = checkpoint.get("best_val")
        self.step = checkpoint["step"]

    def _batch(self, planes, labels):
        planes = planes.to(self.device, non_blocking=True).float()
        planes = planes.contiguous(memory_format=torch.channels_last)
        return planes, labels.to(self.device, non_blocking=True)

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        loss_sum = torch.zeros((), device=self.device)
        top1 = torch.zeros((), device=self.device)
        top5 = torch.zeros((), device=self.device)
        count = 0
        for planes, labels in self.val_loader:
            planes, labels = self._batch(planes, labels)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                logits = self.model(planes)
            loss_sum += F.cross_entropy(logits.float(), labels, reduction="sum")
            top1 += logits.argmax(dim=1).eq(labels).sum()
            top5 += logits.topk(5, dim=1).indices.eq(labels[:, None]).any(dim=1).sum()
            count += labels.numel()
        self.model.train()
        return {
            "val_loss": (loss_sum / count).item(),
            "val_top1": (top1 / count).item(),
            "val_top5": (top5 / count).item(),
            "val_positions": count,
        }

    def _save(self, kind: str, metrics: dict) -> None:
        self.run.save_checkpoint(kind, self.step, self.state(), metrics)

    def train(self) -> None:
        cfg = self.cfg
        total_steps = cfg["total_steps"]
        log_every = cfg.get("log_every", 100)
        eval_every = cfg.get("eval_every", 5000)
        save_every = cfg.get("save_every", 1000)
        snapshot_every = cfg.get("snapshot_every", 50000)
        deadline = time.time() + cfg.get("max_minutes", 1e9) * 60

        self.run.start_session()
        self.model.train()
        loss_sum = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device)
        seen = 0
        window_start = time.time()
        metrics = {}
        reason = "finished"

        try:
            for planes, labels in self.train_loader:
                if self.step >= total_steps:
                    break
                planes, labels = self._batch(planes, labels)

                with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    logits = self.model(planes)
                    loss = F.cross_entropy(logits, labels)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if cfg.get("grad_clip"):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                loss_sum += loss.detach()
                correct += logits.detach().argmax(dim=1).eq(labels).sum()
                seen += labels.numel()
                self.step += 1

                if self.step % log_every == 0:
                    elapsed = time.time() - window_start
                    metrics = {
                        "loss": (loss_sum / log_every).item(),
                        "top1": (correct / seen).item(),
                        "lr": self.scheduler.get_last_lr()[0],
                        "samples_per_s": seen / elapsed,
                    }
                    self.run.log_metrics(self.step, metrics)
                    print(
                        f"step {self.step:>8d}  loss {metrics['loss']:.4f}  "
                        f"top1 {metrics['top1']:.3f}  lr {metrics['lr']:.4g}  "
                        f"{metrics['samples_per_s']:.0f} samples/s",
                        flush=True,
                    )
                    loss_sum.zero_()
                    correct.zero_()
                    seen = 0
                    window_start = time.time()

                if self.step % eval_every == 0 or self.step == total_steps:
                    val = self.validate()
                    metrics = {**metrics, **val}
                    self.run.log_metrics(self.step, val)
                    print(
                        f"step {self.step:>8d}  val_loss {val['val_loss']:.4f}  "
                        f"val_top1 {val['val_top1']:.4f}  val_top5 {val['val_top5']:.4f}",
                        flush=True,
                    )
                    if self.best_val is None or val["val_loss"] < self.best_val["val_loss"]:
                        self.best_val = val
                        self._save("best", metrics)

                if self.step % snapshot_every == 0:
                    self._save("step", metrics)
                if self.step % save_every == 0:
                    self._save("last", metrics)

                if time.time() > deadline:
                    reason = "time limit"
                    break
        except KeyboardInterrupt:
            reason = "interrupted"
        except Exception as e:
            reason = f"error: {type(e).__name__}: {e}"
            raise
        finally:
            self._save("last", metrics)
            if reason.startswith("error"):
                status = "failed"
            else:
                status = "finished" if self.step >= total_steps else "paused"
            self.run.end_session(status, reason)
