import yaml
from pathlib import Path


class Config:
    def __init__(self, data=None):
        if data is None:
            data = {}
        for k, v in data.items():
            self.__setattr__(k, v)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = Config(value)
        elif isinstance(value, list):
            value = [Config(x) if isinstance(x, dict) else x for x in value]
        super().__setattr__(name, value)

    def __getattr__(self, name):
        # auto create nested config if missing
        value = Config()
        super().__setattr__(name, value)
        return value

    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                out[k] = v
        return out

    def save(self, path: str | Path):
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: str | Path):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        return Config(yaml.safe_load(file))
