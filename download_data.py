import sys
import subprocess
from pathlib import Path

from alphago.util.data import load_config


def clone_dataset(name: str, url: str, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {name} from {url} ...")
    result = subprocess.run(
        ["git", "clone", url, str(data_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed:\n{result.stderr.strip()}")
    print(f"Dataset saved to {data_dir}")


def main() -> None:
    main_dir = Path("external/")
    config = load_config(config_path="config/data.yaml")
    datasets = config.datasets

    if not datasets:
        print("No datasets defined in config.")
        sys.exit(0)

    for name, meta in datasets:
        data_dir = main_dir / name

        if data_dir.exists() and any(data_dir.iterdir()):
            print(f"Dataset '{name}' already present at {data_dir}, skipping.")
            continue

        ds_type = meta.type
        url = meta.url

        if not url:
            print(f"No URL specified for '{name}', skipping.")
            continue

        if ds_type != "git":
            print(f"Unsupported type '{ds_type}' for '{name}', skipping.")
            continue

        clone_dataset(name=name, url=url, data_dir=data_dir)


if __name__ == "__main__":
    main()
