import os
import random
import numpy as np

from dlgo.gotypes import Point
from dlgo.goboard import GameState, Move
from dlgo.data.sampling import Sampler
from dlgo.data.process_files import read_txt_files, extract_moves, setup_handicap_game
from dlgo.encoders.base import get_encoder_by_name


class GoDataProcessor:
    def __init__(self, encoder="oneplane", data_directory="go_data"):
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_directory

    def load_go_data(self, data_type="train", num_samples=1000):
        if not os.path.isfile(self.data_dir):
            raise FileNotFoundError(f"Data file not found: {self.data_dir}")

        games = self.read_games_from_file(self.data_dir)
        if num_samples < len(games):
            games = random.sample(games, num_samples)

        features_and_labels = self.process_games(games)
        return features_and_labels

    def process_games(self, games):
        features_and_labels = []

        for i, sgf_content in enumerate(games):
            try:
                parsed_game = self.parse_sgf_game(sgf_content)
                features_and_labels.append(parsed_game)

                if i % 100 == 0:
                    print(f"Processed {i} games...")

            except Exception as e:
                print(f"Error processing game {i}: {e}")
                continue

        return features_and_labels

    def parse_sgf_game(self, sgf_content):
        board_size = 19
        game_state = GameState.new_game(board_size)

        features = []
        labels = []

        moves, handicap_info = extract_moves(sgf_content)
        if handicap_info["is_handicap_game"]:
            # setup game with handicap
            game, handicap_moves = setup_handicap_game(
                board_size, handicap_info)

        for move in moves:
            color, (row, col) = move
            point = Point(row, col)

            board_tensor = self.encoder.encode(game_state)
            move_index = self.encoder.encode_point(point)

            label = np.zeros(self.encoder.num_points())
            label[move_index] = 1

            features.append(board_tensor)
            labels.append(label)

            move_obj = Move.play(point)
            game = game.apply_move(move_obj)
            return {
                "features": np.array(features),
                "labels": np.array(labels),
                "num_moves": len(moves),
                "board_size": board_size,
            }
