from pathlib import Path

import torch
import torch.nn as nn


def normalize_ddp_keys(state_dict: dict, expect_ddp: bool = False) -> dict:
    new_sd = {}

    for k, v in state_dict.items():
        original = k
        while k.startswith("module."):
            k = k[len("module.") :]

        # If expecting DDP, re-add a single module. prefix
        if expect_ddp and original.startswith("module."):
            k = f"module.{k}"

        new_sd[k] = v

    return new_sd


def load_model(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    state_dict = normalize_ddp_keys(state_dict, expect_ddp=False)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
