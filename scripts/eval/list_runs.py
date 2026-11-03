import argparse
from pathlib import Path

from alphago.paths import CHECKPOINT_DIR
from alphago.runs import find_runs


def main():
    parser = argparse.ArgumentParser(description="list training runs and their checkpoints")
    parser.add_argument("--root", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    runs = find_runs(args.root)
    if not runs:
        print(f"no runs under {args.root}")
        return

    header = f"{'stage':<11} {'run':<36} {'status':<9} {'step':>8} {'best top1':>9} {'top5':>6}  commit"
    print(header)
    print("-" * len(header))
    for run in runs:
        info = run.info
        best = (info.get("best") or {}).get("metrics", {})
        top1 = f"{best['val_top1']:.3f}" if "val_top1" in best else "-"
        top5 = f"{best['val_top5']:.3f}" if "val_top5" in best else "-"
        git = info.get("git") or {}
        commit = (git.get("commit") or "?")[:8] + ("+dirty" if git.get("dirty") else "")
        print(f"{info['stage']:<11} {info['run_id']:<36} {info['status']:<9} "
              f"{info['step']:>8} {top1:>9} {top5:>6}  {commit}")
        if args.verbose:
            data = info["config"].get("data_manifest") or {}
            print(f"    model {info['config']['model']['class']}  encoder {info['config']['encoder']['name']}"
                  f"  data {data.get('name')}  parent {info.get('parent')}")
            for name, meta in info["checkpoints"].items():
                print(f"    {name:<22} step {meta['step']:>8}  {meta['saved']}")


if __name__ == "__main__":
    main()
