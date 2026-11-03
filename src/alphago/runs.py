import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from .encoders import get_encoder_by_name
from .networks import AlphaGo, SmallNetwork, SmallResidualNetwork
from .paths import CHECKPOINT_DIR
from .util.model import normalize_ddp_keys
from .util.provenance import git_info, host_info, now

CHECKPOINT_FORMAT = 1
NETWORKS = {
    "AlphaGo": AlphaGo,
    "SmallNetwork": SmallNetwork,
    "SmallResidualNetwork": SmallResidualNetwork,
}


def build_network(spec: dict):
    return NETWORKS[spec["class"]](**spec["kwargs"])


def build_encoder(spec: dict):
    return get_encoder_by_name(spec["name"], spec["board_size"], **spec.get("kwargs", {}))


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


class Run:
    """
    one training run = one directory under checkpoints/<stage>/
      run.json      config, provenance, sessions, checkpoints, best metrics
      metrics.jsonl one line per logged step
      *.pth         self-describing checkpoints
    """

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.info = json.loads((self.dir / "run.json").read_text())

    @classmethod
    def create(
        cls,
        stage: str,
        name: str,
        config: dict,
        parent: Optional[str] = None,
        root: Path = CHECKPOINT_DIR,
    ) -> "Run":
        run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
        run_dir = Path(root) / stage / run_id
        run_dir.mkdir(parents=True)
        info = {
            "run_id": run_id,
            "stage": stage,
            "name": name,
            "created": now(),
            "status": "created",
            "parent": parent,
            "git": git_info(),
            "config": config,
            "sessions": [],
            "checkpoints": {},
            "best": None,
            "step": 0,
        }
        _write_json(run_dir / "run.json", info)
        return cls(run_dir)

    @property
    def run_id(self) -> str:
        return self.info["run_id"]

    def save(self) -> None:
        _write_json(self.dir / "run.json", self.info)

    def start_session(self) -> None:
        self.info["sessions"].append(
            {"started": now(), "start_step": self.info["step"], "git": git_info(), "host": host_info()}
        )
        self.info["status"] = "running"
        self.save()

    def end_session(self, status: str, reason: str) -> None:
        session = self.info["sessions"][-1]
        session.update({"ended": now(), "end_step": self.info["step"], "reason": reason})
        self.info["status"] = status
        self.save()

    def log_metrics(self, step: int, metrics: dict) -> None:
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "time": now(), **metrics}) + "\n")

    def save_checkpoint(self, kind: str, step: int, state: dict, metrics: dict) -> Path:
        """kind is 'last', 'best' or 'step'; 'step' keeps a permanent snapshot"""
        filename = f"step_{step:08d}.pth" if kind == "step" else f"{kind}.pth"
        path = self.dir / filename
        payload = {
            "format": CHECKPOINT_FORMAT,
            "run_id": self.run_id,
            "stage": self.info["stage"],
            "step": step,
            "model": self.info["config"]["model"],
            "encoder": self.info["config"]["encoder"],
            "data": self.info["config"].get("data_manifest"),
            "metrics": metrics,
            **state,
        }
        tmp = path.with_suffix(".pth.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

        self.info["step"] = step
        self.info["checkpoints"][filename] = {"step": step, "saved": now(), "metrics": metrics}
        if kind == "best":
            self.info["best"] = {"step": step, "file": filename, "metrics": metrics}
        self.save()
        return path


def load_checkpoint(path: Path, device="cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def load_policy(path: Path, device="cpu"):
    """returns (model in eval mode, encoder, checkpoint dict) from a self-describing checkpoint"""
    checkpoint = load_checkpoint(path, device)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a run checkpoint (format {checkpoint.get('format')})")
    model = build_network(checkpoint["model"])
    model.load_state_dict(normalize_ddp_keys(checkpoint["model_state_dict"]))
    model.to(device).eval()
    return model, build_encoder(checkpoint["encoder"]), checkpoint


def find_runs(root: Path = CHECKPOINT_DIR):
    return [Run(p.parent) for p in sorted(Path(root).glob("*/*/run.json"))]


def best_checkpoint(stage: str = "supervised", root: Path = CHECKPOINT_DIR) -> Optional[Path]:
    """best.pth of the run in a stage with the highest validation top-1 accuracy"""
    scored = [
        (run.info["best"]["metrics"]["val_top1"], run.dir / run.info["best"]["file"])
        for run in find_runs(root)
        if run.info["stage"] == stage and run.info.get("best")
    ]
    return max(scored)[1] if scored else None
