import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from alphago.go import gotypes, goboard
from alphago.encoders import Encoder
from alphago.agent.policy_agent import (
    PolicyAgent,
    ExperienceCollector,
    ExperienceBuffer,
)
from alphago.paths import CHECKPOINT_DIR
from alphago.runs import Run, best_checkpoint, build_encoder, build_network, load_policy


@torch.no_grad()
def play_self_play_games(model: nn.Module, encoder: Encoder, device: torch.device, gamma: float, num_games: int, amp: bool = False):
    """plays num_games in lockstep with one batched forward pass per move; yields each game when it ends"""
    players = (gotypes.Player.black, gotypes.Player.white)
    games = [goboard.GameState.new_game(encoder.shape()[1]) for _ in range(num_games)]
    collectors = [{p: ExperienceCollector() for p in players} for _ in range(num_games)]
    bots = []
    for game_collectors in collectors:
        game_bots = {}
        for player, collector in game_collectors.items():
            collector.begin_episode()
            game_bots[player] = PolicyAgent(model=model, encoder=encoder, device=device)
            game_bots[player].set_collector(collector)
        bots.append(game_bots)
    moves = [0] * num_games
    active = list(range(num_games))

    while active:
        encoded = [encoder.encode(games[i]) for i in active]
        batch = torch.from_numpy(np.stack(encoded)).to(device, dtype=torch.float32)
        with torch.autocast(device.type, enabled=amp):
            logits = model(batch)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        still_active = []
        for i, state, p in zip(active, encoded, probs):
            game = games[i]
            move = bots[i][game.next_player].select_move_from_probs(game, state, p)
            games[i] = game.apply_move(move)
            moves[i] += 1
            if games[i].is_over():
                yield _finish_game(games[i], collectors[i], moves[i], gamma)
            else:
                still_active.append(i)
        active = still_active


def _finish_game(game, collectors: dict, total_moves: int, gamma: float) -> dict:
    winner = game.winner()
    experiences = []
    for player, collector in collectors.items():
        collector.complete_episode(1.0 if winner == player else -1.0, gamma=gamma)
        buffer = collector.to_buffer()
        experiences.append((buffer.states.astype(np.int8), buffer.actions, buffer.rewards))
    return {"winner": winner, "total_moves": total_moves, "experiences": experiences}


def self_play_worker(model_spec: dict, encoder_spec: dict, device: str, gamma: float, amp: bool, tasks, results):
    torch.set_num_threads(1)
    model = build_network(model_spec).to(device).eval()
    encoder = build_encoder(encoder_spec)
    while True:
        task = tasks.get()
        if task is None:
            return
        state_dict, num_games = task
        model.load_state_dict(state_dict)
        for result in play_self_play_games(model, encoder, torch.device(device), gamma, num_games, amp):
            results.put(result)


class SelfPlayTrainer:
    def __init__(
        self,
        model: nn.Module,
        encoder: Encoder,
        device: torch.device,
        run: Run,
        start_iteration: int = 0,
        gamma: float = 1.0,
        workers: int = 0,
        worker_devices: list[str] | None = None,
        amp: bool = False,
    ):
        self.model = model.to(device)
        self.encoder = encoder
        self.device = device
        self.run = run
        self.board_size = encoder.shape()[1]
        self.iteration = start_iteration
        self.gamma = gamma
        self.amp = amp and device.type == "cuda"
        self.skipped_batches = 0
        self.workers = []
        if workers:
            ctx = mp.get_context("spawn")
            self.tasks, self.results = ctx.Queue(), ctx.Queue()
            devices = worker_devices or [str(device)]
            config = run.info["config"]
            for i in range(workers):
                p = ctx.Process(
                    target=self_play_worker,
                    args=(config["model"], config["encoder"], devices[i % len(devices)], gamma, self.amp, self.tasks, self.results),
                    daemon=True,
                )
                p.start()
                self.workers.append(p)

    def close(self):
        for _ in self.workers:
            self.tasks.put(None)
        for p in self.workers:
            p.join()

    def _play_games(self, num_games: int):
        if not self.workers:
            yield from play_self_play_games(self.model, self.encoder, self.device, self.gamma, num_games, self.amp)
            return
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        base, extra = divmod(num_games, len(self.workers))
        for i in range(len(self.workers)):
            if base + (i < extra):
                self.tasks.put((state_dict, base + (i < extra)))
        for _ in range(num_games):
            yield self.results.get()

    def collect_self_play_experiences(self, num_games: int) -> tuple[ExperienceBuffer, dict]:
        self.model.eval()
        states, actions, rewards = [], [], []
        black_wins, game_lengths = 0, []

        for result in tqdm(self._play_games(num_games), total=num_games, desc="Self-play"):
            black_wins += result["winner"] == gotypes.Player.black
            game_lengths.append(result["total_moves"])
            for s, a, r in result["experiences"]:
                states.append(s)
                actions.append(a)
                rewards.append(r)

        experience = ExperienceBuffer(
            states=np.concatenate(states),
            actions=np.concatenate(actions),
            rewards=np.concatenate(rewards),
        )
        stats = {
            "black_win_rate": black_wins / num_games,
            "avg_game_length": float(np.mean(game_lengths)),
        }
        return experience, stats

    def train_on_experience(
        self,
        experience: ExperienceBuffer,
        learning_rate: float,
        clipnorm: float | None,
        batch_size: int,
    ) -> float:
        self.model.train()

        played = experience.actions >= 0
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(experience.states[played]),
            torch.from_numpy(experience.actions[played].astype(np.int64)),
            torch.from_numpy(experience.rewards[played].astype(np.float32)),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        use_amp = self.amp
        scaler = torch.amp.GradScaler(enabled=use_amp)
        steps, skipped = 0, 0

        total_loss = 0.0
        for batch_states, batch_actions, batch_rewards in loader:
            batch_states = batch_states.to(self.device, dtype=torch.float32)
            batch_actions = batch_actions.to(self.device)
            batch_rewards = batch_rewards.to(self.device)

            with torch.autocast(self.device.type, enabled=use_amp):
                logits = self.model(batch_states)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            chosen = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            loss = -(batch_rewards * chosen).mean()
            if not torch.isfinite(loss):
                skipped += 1
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if clipnorm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), clipnorm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            steps += 1

        self.skipped_batches = skipped
        return total_loss / max(steps, 1)

    def save_checkpoint(self, metrics: dict, snapshot: bool):
        state = {"model_state_dict": self.model.state_dict(), "iteration": self.iteration}
        self.run.save_checkpoint("last", self.iteration, state, metrics)
        if snapshot:
            self.run.save_checkpoint("step", self.iteration, state, metrics)

    def train(
        self,
        num_iterations: int,
        games_per_iteration: int,
        learning_rate: float,
        clipnorm: float | None,
        batch_size: int,
        snapshot_every: int,
        max_minutes: float | None = None,
    ):
        print(f"Run:                    {self.run.dir}")
        print(f"Iterations:             {num_iterations}")
        print(f"Games per iteration:    {games_per_iteration}")
        print(f"Learning rate:          {learning_rate}")
        print(f"Batch size:             {batch_size}")
        print(f"Device:                 {self.device}")

        self.run.start_session()
        end = self.iteration + num_iterations
        started = time.monotonic()
        last_duration = 0.0
        reason = "iterations done"
        try:
            while self.iteration < end:
                elapsed = time.monotonic() - started
                if max_minutes is not None and elapsed + last_duration > max_minutes * 60:
                    reason = "time budget"
                    break
                iteration_start = time.monotonic()
                print(f"\nIteration {self.iteration + 1}/{end}")
                experience, stats = self.collect_self_play_experiences(games_per_iteration)
                print(
                    f"{len(experience.states)} positions, black win rate {stats['black_win_rate']:.1%}, "
                    f"avg length {stats['avg_game_length']:.0f}"
                )

                loss = self.train_on_experience(experience, learning_rate, clipnorm, batch_size)
                print(f"loss: {loss:.6f}, skipped non-finite batches: {self.skipped_batches}")

                self.iteration += 1
                last_duration = time.monotonic() - iteration_start
                metrics = {
                    "loss": loss,
                    "positions": len(experience.states),
                    "skipped_batches": self.skipped_batches,
                    "seconds": last_duration,
                    **stats,
                }
                self.run.log_metrics(self.iteration, metrics)
                self.save_checkpoint(metrics, snapshot=self.iteration % snapshot_every == 0)
        except KeyboardInterrupt:
            self.run.end_session("paused", "interrupted")
            raise
        finally:
            self.close()
        self.run.end_session("paused", reason)
        print(f"Self-play training stopped: {reason}")


def parse_args():
    parser = argparse.ArgumentParser(description="policy gradient self-play from a policy checkpoint")
    parser.add_argument(
        "--init-from",
        type=Path,
        help="policy checkpoint to start from (default: best supervised run)",
    )
    parser.add_argument("--resume", type=Path, help="rl run directory to continue")
    parser.add_argument("--name", default="policy_gradient")
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--games_per_iteration", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clipnorm", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--snapshot_every", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0, help="reward discount per move")
    parser.add_argument("--workers", type=int, default=0, help="self-play processes (0 = play in the main process)")
    parser.add_argument("--worker-devices", nargs="*", help="devices the workers cycle through (default: --device)")
    parser.add_argument("--amp", choices=["on", "off"], default="off", help="fp16 autocast (needs tensor cores, e.g. T4)")
    parser.add_argument("--max-minutes", type=float, help="stop before an iteration that would exceed this")
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.resume:
        run = Run(args.resume)
        model, encoder, checkpoint = load_policy(run.dir / "last.pth", device)
        start_iteration = checkpoint["iteration"]
        args.gamma = run.info["config"]["rl"].get("gamma", args.gamma)
    else:
        init_path = args.init_from or best_checkpoint(root=args.checkpoint_root)
        if init_path is None:
            raise SystemExit("no supervised checkpoint found, pass --init-from")
        model, encoder, checkpoint = load_policy(init_path, device)
        config = {
            "stage": "rl",
            "name": args.name,
            "model": checkpoint["model"],
            "encoder": checkpoint["encoder"],
            "init_from": {"run_id": checkpoint["run_id"], "step": checkpoint["step"], "file": Path(init_path).name},
            "rl": {
                "learning_rate": args.learning_rate,
                "clipnorm": args.clipnorm,
                "batch_size": args.batch_size,
                "games_per_iteration": args.games_per_iteration,
                "gamma": args.gamma,
                "amp": args.amp,
            },
        }
        run = Run.create("rl", args.name, config, parent=checkpoint["run_id"], root=args.checkpoint_root)
        start_iteration = 0

    print(f"Loaded {checkpoint['model']['class']} ({encoder.name()} encoder, {encoder.num_planes} planes) "
          f"from {checkpoint['run_id']} step {checkpoint['step']}")

    trainer = SelfPlayTrainer(
        model, encoder, device, run, start_iteration,
        gamma=args.gamma, workers=args.workers, worker_devices=args.worker_devices, amp=args.amp == "on",
    )
    trainer.train(
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        learning_rate=args.learning_rate,
        clipnorm=args.clipnorm,
        batch_size=args.batch_size,
        snapshot_every=args.snapshot_every,
        max_minutes=args.max_minutes,
    )


if __name__ == "__main__":
    main()
