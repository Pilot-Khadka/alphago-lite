import os
from pathlib import Path

ROOT = Path(os.environ.get("ALPHAGO_ROOT", Path(__file__).resolve().parents[2]))

CONFIG_DIR = ROOT / "config"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPERIENCE_DIR = DATA_DIR / "experience"
PRO_GAMES_DIR = RAW_DIR / "computer-go-dataset" / "Professional"

CHECKPOINT_DIR = ROOT / "checkpoints"
SUPERVISED_DIR = CHECKPOINT_DIR / "supervised"
RL_DIR = CHECKPOINT_DIR / "rl"

RUNS_DIR = ROOT / "runs"
