from typing import Optional

import time
import multiprocessing as mp
from collections import defaultdict

import torch
import numpy as np
import torch.nn as nn

from alphago.go import gotypes, goboard
from alphago.encoders import OnePlaneEncoder, Encoder
from alphago.agent.policy_agent import ExperienceCollector, ExperienceBuffer


class InferenceServer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        request_queue: mp.Queue,
        response_queues: dict[int, mp.Queue],
        max_batch_size: int = 256,
        batch_wait_seconds: float = 0.005,
    ):
        self.model = model
        self.device = device
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.max_batch_size = max_batch_size
        self.batch_wait_seconds = batch_wait_seconds

    def serve(self, total_requests: int):
        self.model.eval()
        processed = 0

        with torch.no_grad():
            while processed < total_requests:
                requests = self._drain_queue()
                if not requests:
                    continue

                worker_ids, game_ids, states = zip(*requests)
                batch = torch.tensor(
                    np.stack(states), dtype=torch.float32, device=self.device
                )
                q_batch = self.model(batch).cpu().numpy()

                for worker_id, game_id, q_values in zip(worker_ids, game_ids, q_batch):
                    self.response_queues[worker_id].put((game_id, q_values))

                processed += len(requests)

    def _drain_queue(self) -> list[tuple]:
        requests = []
        deadline = time.monotonic() + self.batch_wait_seconds

        while len(requests) < self.max_batch_size:
            try:
                item = self.request_queue.get_nowait()
                requests.append(item)
            except Exception:
                if requests or time.monotonic() >= deadline:
                    break
                time.sleep(0.0005)

        return requests


def _select_action(
    q_values: np.ndarray, game, encoder, temperature: float
) -> goboard.Move:
    board_size = game.board.size
    num_moves = board_size * board_size  # all points + pass

    candidates = []
    for idx in range(num_moves - 1):
        point = encoder.decode_point_index(idx)
        if game.is_valid_move(goboard.Move.play(point)):
            candidates.append(idx)
    candidates.append(num_moves - 1)  # pass is always legal

    candidate_q = q_values[candidates]

    if temperature < 1e-4:
        chosen = candidates[int(np.argmax(candidate_q))]
    else:
        shifted = candidate_q - candidate_q.max()
        probs = np.exp(shifted / temperature)
        probs /= probs.sum()
        chosen = candidates[int(np.random.choice(len(candidates), p=probs))]

    if chosen == num_moves - 1:
        return goboard.Move.pass_turn()
    return goboard.Move.play(encoder.decode_point_index(chosen))


def _worker_fn(
    worker_id: int,
    num_games: int,
    board_size: int,
    temperature: float,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
    encoder: Encoder,
):
    games = [goboard.GameState.new_game(board_size) for _ in range(num_games)]
    collectors = [ExperienceCollector() for _ in range(num_games)]
    winners = [None] * num_games

    for c in collectors:
        c.begin_episode()

    active = set(range(num_games))

    while active:
        active_list = list(active)

        for gid in active_list:
            state = encoder.encode(games[gid])
            request_queue.put((worker_id, gid, state))

        responses: dict[int, np.ndarray] = {}
        for _ in active_list:
            gid, q_values = response_queue.get()
            responses[gid] = q_values

        finished_this_round = []
        for gid in active_list:
            move = _select_action(responses[gid], games[gid], encoder, temperature)
            action_idx = encoder.encode_point(move.point) if move.is_play else -1

            collectors[gid].record_decision(
                state=encoder.encode(games[gid]),
                action=action_idx,
            )

            games[gid] = games[gid].apply_move(move)

            if games[gid].is_over():
                # pyrefly: ignore
                winners[gid] = games[gid].winner()
                finished_this_round.append(gid)

        active -= set(finished_this_round)

    buffers = []
    for gid in range(num_games):
        reward = 1.0 if winners[gid] == gotypes.Player.black else -1.0
        collectors[gid].complete_episode(reward)
        buffers.append((winners[gid], collectors[gid].to_buffer()))

    for gid, (winner, buffer) in enumerate(
        zip(winners, [c.to_buffer() for c in collectors])
    ):
        bad = (buffer.actions >= board_size * board_size) | (
            (buffer.actions < 0) & (buffer.actions != -1)
        )
        if bad.any():
            raise ValueError(
                f"Game {gid}: illegal action indices {buffer.actions[bad]}"
            )
    result_queue.put(buffers)


class ParallelSelfPlay:
    def __init__(
        self,
        model: nn.Module,
        encoder: Encoder,
        device: torch.device,
        board_size: int,
        temperature: float,
        num_workers: int | None,
        games_per_worker: int = 8,
        max_batch_size: int = 256,
    ):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.board_size = board_size
        self.temperature = temperature
        self.num_workers = num_workers or mp.cpu_count()
        self.games_per_worker = games_per_worker
        self.max_batch_size = max_batch_size

    def collect(self, num_games: int) -> tuple[ExperienceBuffer, dict]:
        actual_workers = min(self.num_workers, num_games)
        base = num_games // actual_workers
        extras = num_games % actual_workers
        games_per_worker = [
            base + (1 if i < extras else 0) for i in range(actual_workers)
        ]

        request_queue = mp.Queue()
        response_queues = {i: mp.Queue() for i in range(actual_workers)}
        result_queue = mp.Queue()

        workers = []
        for worker_id, n_games in enumerate(games_per_worker):
            p = mp.Process(
                target=_worker_fn,
                args=(
                    worker_id,
                    n_games,
                    self.board_size,
                    self.temperature,
                    request_queue,
                    response_queues[worker_id],
                    result_queue,
                    self.encoder,
                ),
                daemon=True,
            )
            p.start()
            workers.append(p)

        all_results = []
        self.model.eval()

        with torch.no_grad():
            while len(all_results) < actual_workers:
                requests = self._drain_queue(request_queue)
                if requests:
                    worker_ids, game_ids, states = zip(*requests)
                    batch = torch.tensor(
                        np.stack(states), dtype=torch.float32, device=self.device
                    )
                    q_batch = self.model(batch).cpu().numpy()
                    for wid, gid, q_values in zip(worker_ids, game_ids, q_batch):
                        response_queues[wid].put((gid, q_values))

                # Check if any workers have finished
                while True:
                    try:
                        result = result_queue.get_nowait()
                        all_results.append(result)
                    except Exception:
                        break

        for p in workers:
            p.join()

        return self._merge_results(all_results)

    def _drain_queue(self, queue: mp.Queue, max_items: Optional[int] = None) -> list:
        limit = max_items or self.max_batch_size
        items = []
        deadline = time.monotonic() + 0.005
        while len(items) < limit:
            try:
                items.append(queue.get_nowait())
            except Exception:
                if items or time.monotonic() >= deadline:
                    break
                time.sleep(0.0005)
        return items

    def _merge_results(self, all_results: list) -> tuple[ExperienceBuffer, dict]:
        all_states, all_actions, all_rewards = [], [], []
        stats = defaultdict(lambda: {"total_reward": 0.0, "num_moves": 0})

        for worker_results in all_results:
            for winner, buffer in worker_results:
                all_states.append(buffer.states)
                all_actions.append(buffer.actions)
                all_rewards.append(buffer.rewards)

                for player in [gotypes.Player.black, gotypes.Player.white]:
                    reward = 1.0 if winner == player else -1.0
                    stats[player]["total_reward"] += reward
                    stats[player]["num_moves"] += len(buffer.rewards) // 2

        combined = ExperienceBuffer(
            states=np.concatenate(all_states),
            actions=np.concatenate(all_actions),
            rewards=np.concatenate(all_rewards),
        )

        stats_summary = {
            player: {
                "total_moves": s["num_moves"],
                "total_reward": s["total_reward"],
                "avg_reward": s["total_reward"] / s["num_moves"]
                if s["num_moves"]
                else 0.0,
            }
            for player, s in stats.items()
        }

        return combined, stats_summary
