import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from alphago.data.games import InvalidGame, parse_sgf, replay, save_games
from alphago.paths import PROCESSED_DIR, PRO_GAMES_DIR, ROOT
from alphago.util.provenance import git_info, now, sha256_file

SPLITS = (("train", 960), ("val", 980), ("test", 1000))


def check_game(line):
    try:
        record = parse_sgf(line)
        if record.board_size != 19:
            return None, "not 19x19"
        if len(record.moves) < 20:
            return None, "fewer than 20 moves"
        for _ in replay(record):
            pass
        return record, None
    except (InvalidGame, ValueError, AssertionError) as e:
        return None, f"invalid: {type(e).__name__}"


def split_of(key):
    bucket = int(key[:8], 16) % 1000
    return next(name for name, upper in SPLITS if bucket < upper)


def main():
    parser = argparse.ArgumentParser(description="parse raw SGF games into a game archive")
    parser.add_argument("--name", default="pro_v1")
    parser.add_argument("--sources", nargs="+", type=Path,
                        default=sorted(PRO_GAMES_DIR.glob("*.txt")))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = PROCESSED_DIR / "games" / args.name
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; pick a new --name")

    lines, sources = [], []
    for path in args.sources:
        game_lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        lines.extend(game_lines)
        sources.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "games": len(game_lines),
        })

    with Pool(args.workers) as pool:
        results = pool.map(check_game, lines, chunksize=200)

    skipped = Counter(reason for _, reason in results if reason)
    splits = {name: [] for name, _ in SPLITS}
    seen = set()
    duplicates = 0
    for record, _ in results:
        if record is None:
            continue
        key = record.key()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        splits[split_of(key)].append(record)

    out_dir.mkdir(parents=True)
    split_info = {}
    for name, records in splits.items():
        path = out_dir / f"{name}.npz"
        save_games(path, records)
        split_info[name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "games": len(records),
            "positions": sum(len(r.moves) for r in records),
        }

    manifest = {
        "name": args.name,
        "created": now(),
        "git": git_info(),
        "sources": sources,
        "games_read": len(lines),
        "skipped": dict(skipped),
        "duplicates": duplicates,
        "split_rule": "sha1(game) bucket: train <960, val <980, test <1000 (of 1000)",
        "splits": split_info,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("games_read", "skipped", "duplicates", "splits")}, indent=2))


if __name__ == "__main__":
    main()
