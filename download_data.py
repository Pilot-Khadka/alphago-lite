import sys
import yaml
import subprocess
from pathlib import Path


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open() as f:
        return yaml.safe_load(f)


def dataset_exists(name: Path) -> bool:
    target = external_dir / name
    return target.exists() and any(target.iterdir())


def clone_dataset(name: str, url: str) -> None:
    external_dir.mkdir(parents=True, exist_ok=True)
    target = external_dir / name
    print(f"Cloning {name} from {url} ...")
    result = subprocess.run(
        ["git", "clone", url, str(target)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed:\n{result.stderr.strip()}")
    print(f"Done — dataset saved to {target}")


def main() -> None:
    config = load_config()
    datasets: dict = config.get("datasets", {})

    if not datasets:
        print("No datasets defined in config.")
        sys.exit(0)

    for name, meta in datasets.items():
        if dataset_exists(name):
            print(f"Dataset '{name}' already present at external/{name}, skipping.")
            continue

        ds_type = meta.get("type", "git")
        url = meta.get("url")

        if not url:
            print(f"No URL specified for '{name}', skipping.")
            continue

        if ds_type != "git":
            print(f"Unsupported type '{ds_type}' for '{name}', skipping.")
            continue

        clone_dataset(name, url)


if __name__ == "__main__":
    main()
