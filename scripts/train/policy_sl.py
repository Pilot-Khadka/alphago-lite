import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from alphago.data.stream import PositionStream
from alphago.paths import CHECKPOINT_DIR, CONFIG_DIR, ROOT
from alphago.runs import Run, build_network, load_checkpoint
from alphago.train.policy_trainer import PolicyTrainer
from alphago.util.model import normalize_ddp_keys


def parse_args():
    parser = argparse.ArgumentParser(
        description="supervised policy training from game records"
    )
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "sl_alphago.yaml")
    parser.add_argument("--resume", type=Path, help="run directory to continue")
    parser.add_argument(
        "--init-from",
        type=Path,
        help="checkpoint whose weights start a new run (fresh optimizer and lr schedule)",
    )
    parser.add_argument(
        "--games", type=Path, help="game archive directory (overrides config)"
    )
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--amp", choices=["on", "off"])
    parser.add_argument("--name", help="run name (overrides config)")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override config values, e.g. train.eval_every=500",
    )
    return parser.parse_args()


def apply_overrides(config, overrides):
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            node = node[part]
        node[leaf] = yaml.safe_load(value)


def make_loaders(config, games_dir, workers):
    data, encoder = config["data"], config["encoder"]
    common = dict(
        encoder_name=encoder["name"],
        encoder_kwargs=encoder.get("kwargs", {}),
        include_pass=data.get("include_pass", False),
    )
    train = PositionStream(
        games_dir / "train.npz",
        augment=data.get("augment", True),
        shuffle_buffer=data.get("shuffle_buffer", 20000),
        seed=data.get("seed", 0),
        **common,
    )
    val = PositionStream(
        games_dir / "val.npz",
        augment=False,
        shuffle=False,
        loop=False,
        sample_every=data.get("val_sample_every", 8),
        **common,
    )
    batch_size = config["train"]["batch_size"]
    loader_kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    if workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    return (
        DataLoader(train, **loader_kwargs),
        DataLoader(val, batch_size=batch_size, num_workers=workers, pin_memory=True),
    )


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.resume and args.init_from:
        raise SystemExit("use either --resume or --init-from")
    run = Run(args.resume) if args.resume else None
    config = run.info["config"] if run else yaml.safe_load(args.config.read_text())

    if not args.resume:
        apply_overrides(config, args.set)
        if args.name:
            config["name"] = args.name
    train_cfg = config["train"]
    if args.max_minutes is not None:
        train_cfg["max_minutes"] = args.max_minutes
    if args.total_steps is not None:
        train_cfg["total_steps"] = args.total_steps
    if args.amp is not None:
        train_cfg["amp"] = args.amp == "on"
    workers = (
        args.workers if args.workers is not None else config["data"].get("workers", 4)
    )

    games_dir = args.games or ROOT / config["data"]["games"]
    manifest = json.loads((games_dir / "manifest.json").read_text())
    data_manifest = {
        "name": manifest["name"],
        "splits": {k: v["sha256"] for k, v in manifest["splits"].items()},
    }
    if args.resume and config.get("data_manifest") != data_manifest:
        raise SystemExit(f"{games_dir} is not the dataset this run started with")
    config["data_manifest"] = data_manifest

    init = None
    if args.init_from:
        init = load_checkpoint(args.init_from)
        for key in ("model", "encoder"):
            if init[key] != config[key]:
                raise SystemExit(
                    f"{args.init_from} {key} {init[key]} does not match config {config[key]}"
                )
        config["init_from"] = {
            "run_id": init["run_id"],
            "step": init["step"],
            "file": args.init_from.name,
        }

    if run is None:
        parent = init["run_id"] if init else None
        run = Run.create(
            config["stage"],
            config["name"],
            config,
            parent=parent,
            root=args.checkpoint_root,
        )
    print(f"run {run.run_id} -> {run.dir}", flush=True)

    model = build_network(config["model"])
    if init:
        model.load_state_dict(normalize_ddp_keys(init["model_state_dict"]))
        print(f"initialised from {init['run_id']} step {init['step']}", flush=True)
    train_loader, val_loader = make_loaders(config, games_dir, workers)
    trainer = PolicyTrainer(run, model, train_loader, val_loader, device, train_cfg)

    if args.resume:
        trainer.load_state(load_checkpoint(run.dir / "last.pth", device))
        print(f"resumed at step {trainer.step}", flush=True)

    trainer.train()
    print(
        json.dumps(
            {"run": run.run_id, "step": trainer.step, "best": run.info["best"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
