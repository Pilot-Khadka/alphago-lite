from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List


class AIModelType(Enum):
    NAIVE = "Random"
    MCTS_PURE = "Pure MCTS"
    CNN_POLICY = "CNN Policy Network"
    VALUE_NETWORK = "Value Network"
    ALPHAGO_ZERO = "AlphaGo Zero"
    CUSTOM_MODEL = "Custom Model"


class NetworkArchitecture(Enum):
    RESNET = "ResNet"
    TRANSFORMER = "Transformer"
    EFFICIENTNET = "EfficientNet"
    CUSTOM = "Custom"


@dataclass
class GameAnalysis:
    move_evaluations: List[float]
    policy_distribution: Dict[str, float]
    value_estimate: float
    variation_tree: Dict[str, Any]
    computational_stats: Dict[str, Any]


class GameMode(Enum):
    HUMAN_VS_HUMAN = "Human vs Human"
    HUMAN_VS_AI = "Human vs AI"
    AI_VS_AI = "AI vs AI"
    ANALYSIS_MODE = "Position Analysis"
