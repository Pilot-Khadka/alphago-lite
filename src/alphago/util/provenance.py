import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..paths import ROOT

BUILD_INFO_ENV = "ALPHAGO_BUILD_INFO"


def _git(*args):
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info() -> dict:
    """
    commit + dirty flag of the code that is running
    on machines without the repo (kaggle) this comes from the build info file
    written when the code was packaged
    """
    build_info = os.environ.get(BUILD_INFO_ENV)
    if build_info and Path(build_info).exists():
        return json.loads(Path(build_info).read_text())["git"]

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None, "diff_sha256": None}
    diff = _git("diff", "HEAD") or ""
    untracked = _git("ls-files", "--others", "--exclude-standard") or ""
    return {
        "commit": commit,
        "dirty": bool(diff or untracked),
        "diff_sha256": hashlib.sha256((diff + untracked).encode()).hexdigest()
        if diff or untracked
        else None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def host_info() -> dict:
    info: dict = {"hostname": platform.node(), "python": platform.python_version()}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpus"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return info
