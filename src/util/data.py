import yaml
from pathlib import Path


def load_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
