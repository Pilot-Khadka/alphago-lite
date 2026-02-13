import yaml
from pathlib import Path


class Config:
    def __init__(self, data):
        for k, v in data.items():
            if isinstance(v, dict):
                v = Config(v)
            elif isinstance(v, list):
                v = [Config(x) if isinstance(x, dict) else x for x in v]
            setattr(self, k, v)


def load_config(config_path: str | Path):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        return Config(yaml.safe_load(file))
