from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QGridLayout,
    QLineEdit,
)
from dlgo.configs.types import AIModelType, NetworkArchitecture


class ModelConfigWidget(QWidget):
    """Widget for configuring AI models"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Model Selection
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout()

        model_layout.addWidget(QLabel("Model Type:"), 0, 0)
        self.model_type_combo = QComboBox()
        for model in AIModelType:
            self.model_type_combo.addItem(model.value, model)
        model_layout.addWidget(self.model_type_combo, 0, 1)

        model_layout.addWidget(QLabel("Architecture:"), 1, 0)
        self.arch_combo = QComboBox()
        for arch in NetworkArchitecture:
            self.arch_combo.addItem(arch.value, arch)
        model_layout.addWidget(self.arch_combo, 1, 1)

        model_layout.addWidget(QLabel("Model Path:"), 2, 0)
        self.model_path_edit = QLineEdit()
        model_layout.addWidget(self.model_path_edit, 2, 1)

        # MCTS Configuration
        mcts_group = QGroupBox("MCTS Parameters")
        mcts_layout = QGridLayout()

        mcts_layout.addWidget(QLabel("Simulations:"), 0, 0)
        self.simulations_spin = QSpinBox()
        self.simulations_spin.setRange(100, 100000)
        self.simulations_spin.setValue(1600)
        mcts_layout.addWidget(self.simulations_spin, 0, 1)

        mcts_layout.addWidget(QLabel("C_puct:"), 1, 0)
        self.cpuct_spin = QDoubleSpinBox()
        self.cpuct_spin.setRange(0.1, 5.0)
        self.cpuct_spin.setValue(1.0)
        self.cpuct_spin.setSingleStep(0.1)
        mcts_layout.addWidget(self.cpuct_spin, 1, 1)

        mcts_layout.addWidget(QLabel("Temperature:"), 2, 0)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setValue(1.0)
        self.temperature_spin.setSingleStep(0.1)
        mcts_layout.addWidget(self.temperature_spin, 2, 1)

        mcts_group.setLayout(mcts_layout)
        model_group.setLayout(model_layout)

        layout.addWidget(model_group)
        layout.addWidget(mcts_group)
        layout.addStretch()
        self.setLayout(layout)
