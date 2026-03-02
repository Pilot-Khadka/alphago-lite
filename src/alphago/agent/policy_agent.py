import numpy as np
import torch

from ..go import goboard
from .helpers import is_point_an_eye
from alphago.agent.base import Agent


class ExperienceBuffer:
    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def save(self, path: str):
        torch.save(
            {
                "states": torch.tensor(self.states),
                "actions": torch.tensor(self.actions),
                "rewards": torch.tensor(self.rewards),
            },
            path,
        )

    @staticmethod
    def load(path: str):
        data = torch.load(path)
        return ExperienceBuffer(
            states=data["states"].numpy()
            if torch.is_tensor(data["states"])
            else data["states"],
            actions=data["actions"].numpy()
            if torch.is_tensor(data["actions"])
            else data["actions"],
            rewards=data["rewards"].numpy()
            if torch.is_tensor(data["rewards"])
            else data["rewards"],
        )


class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.current_episode_states = []
        self.current_episode_actions = []

    def begin_episode(self):
        self.current_episode_states = []
        self.current_episode_actions = []

    def record_decision(self, state, action):
        self.current_episode_states.append(state)
        self.current_episode_actions.append(action)

    def complete_episode(self, reward, gamma=0.99):
        num_states = len(self.current_episode_states)
        discounted = np.zeros(num_states)
        discounted[-1] = reward
        for t in reversed(range(num_states - 1)):
            discounted[t] = gamma * discounted[t + 1]
        self.states += self.current_episode_states
        self.actions += self.current_episode_actions
        self.rewards += discounted.tolist()
        self.current_episode_states = []
        self.current_episode_actions = []

    def to_buffer(self):
        return ExperienceBuffer(
            states=np.array(self.states),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
        )


def prepare_experience_data(
    experience: ExperienceBuffer,
    board_width: int,
    board_height: int,
):
    experience_size = experience.actions.shape[0]
    target_vectors = np.zeros((experience_size, board_width * board_height))

    for i in range(experience_size):
        action = experience.actions[i]
        reward = experience.rewards[i]
        target_vectors[i][action] = reward

    return target_vectors


class PolicyAgent(Agent):
    def __init__(self, model, encoder, device: torch.device):
        super().__init__()
        self._model = model
        self._encoder = encoder
        self._device = device
        self.eps = 1e-6
        self.collector = None

    @torch.no_grad()
    def predict(self, encoded_state):
        input_arr = np.array([encoded_state])
        input_tensor = torch.tensor(input_arr, dtype=torch.float32, device=self._device)
        logits = self._model(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy()

    def set_collector(self, collector):
        self.collector = collector

    def select_move(self, game_state):
        encoded_state = self._encoder.encode(game_state)

        num_moves = self._encoder.board_size**2
        move_probs = self.predict(encoded_state).reshape(-1)  # shape (board_size^2,)

        move_probs = move_probs**3
        move_probs = np.clip(move_probs, self.eps, 1 - self.eps)
        move_probs /= np.sum(move_probs)

        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs
        )

        for point_idx in ranked_moves:
            point = self._encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(
                goboard.Move.play(point)
            ) and not is_point_an_eye(game_state.board, point, game_state.next_player):
                if self.collector is not None:
                    self.collector.record_decision(encoded_state, point_idx)
                return goboard.Move.play(point)

        if self.collector is not None:
            self.collector.record_decision(encoded_state, -1)
        return goboard.Move.pass_turn()

    def encode_state(self, game_state) -> np.ndarray:
        return self._encoder.encode(game_state)
