from pathlib import Path

import torch

from alphago.agent.base import Agent
from alphago.agent.helpers import is_point_an_eye
from alphago.encoders import AlphaGoEncoder, OnePlaneEncoder, SevenPlaneEncoder
from alphago.go import Move
from alphago.paths import ROOT
from alphago.networks import AlphaGo, SmallNetwork, SmallResidualNetwork
from alphago.runs import load_policy
from alphago.util.model import normalize_ddp_keys


class PolicyBot(Agent):
    def __init__(self, model, encoder, device, info=None):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.device = device
        self.board_size = encoder.board_size
        self.info = info or {}

    @torch.inference_mode()
    def select_move(self, game_state):
        encoded = self.encoder.encode(game_state)
        x = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
        logits = self.model(x.unsqueeze(0))[0, : self.encoder.num_points()]

        for idx in torch.argsort(logits, descending=True).tolist():
            point = self.encoder.decode_point_index(idx)
            move = Move.play(point)
            if not is_point_an_eye(
                game_state.board, point, game_state.next_player
            ) and game_state.is_valid_move(move):
                return move
        return Move.pass_turn()


def _num_blocks(state_dict, prefix):
    return len({k.split(".")[1] for k in state_dict if k.startswith(prefix)})


def _board_size(num_points):
    return int(round(num_points**0.5))


def build_model(state_dict):
    if "conv_in.weight" in state_dict:
        in_channels = state_dict["conv_in.weight"].shape[1]
        board_size = _board_size(state_dict["policy_fc.weight"].shape[0])
        model = SmallResidualNetwork(
            channel_size=in_channels,
            board_size=board_size,
            num_blocks=_num_blocks(state_dict, "res_blocks."),
        )
    elif "backbone.0.0.weight" in state_dict:
        if "head.6.weight" in state_dict:
            raise ValueError("checkpoint is a value network, not a policy network")
        first = state_dict["backbone.0.0.weight"]
        in_channels = first.shape[1]
        board_size = _board_size(state_dict["head.4.weight"].shape[1] // 2)
        model = AlphaGo(
            board_size=board_size,
            in_channels=in_channels,
            num_filters=first.shape[0],
            num_layers=_num_blocks(state_dict, "backbone."),
            first_kernel_size=first.shape[2],
        )
    elif "features.0.weight" in state_dict:
        in_channels = state_dict["features.0.weight"].shape[1]
        board_size = _board_size(state_dict["classifier.1.weight"].shape[0])
        model = SmallNetwork(in_channels, board_size)
    else:
        raise ValueError("unrecognised checkpoint architecture")
    return model, in_channels, board_size


def build_encoder(in_channels, board_size):
    if in_channels == 1:
        return OnePlaneEncoder(board_size)
    if in_channels == 7:
        return SevenPlaneEncoder(board_size)
    if in_channels in (48, 49):
        return AlphaGoEncoder(board_size, use_player_plane=in_channels == 49)
    raise ValueError(f"no encoder with {in_channels} planes")


def _display_path(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_policy_bot(checkpoint_path: Path, device: torch.device) -> PolicyBot:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    info: dict = {"file": _display_path(checkpoint_path)}
    if checkpoint.get("format") is not None:
        model, encoder, _ = load_policy(checkpoint_path, device)
        metrics = checkpoint.get("metrics") or {}
        info.update(
            run_id=checkpoint["run_id"],
            step=checkpoint["step"],
            model=checkpoint["model"]["class"],
            encoder=checkpoint["encoder"]["name"],
            val_top1=metrics.get("val_top1"),
            val_top5=metrics.get("val_top5"),
        )
        return PolicyBot(model, encoder, device, info)

    state_dict = normalize_ddp_keys(checkpoint.get("model_state_dict", checkpoint))

    model, in_channels, board_size = build_model(state_dict)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    encoder = build_encoder(in_channels, board_size)
    info.update(model=type(model).__name__, encoder=encoder.name())
    return PolicyBot(model, encoder, device, info)
