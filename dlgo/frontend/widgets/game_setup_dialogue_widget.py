from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGridLayout,
    QSlider,
)
from dlgo.configs.types import GameMode
from dlgo.configs.types import AIModelType


class GameSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Game Setup")
        self.setModal(True)
        self.setMinimumSize(500, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # game Mode Selection
        mode_group = QGroupBox("Game Mode")
        mode_layout = QVBoxLayout()

        self.mode_combo = QComboBox()
        for mode in GameMode:
            self.mode_combo.addItem(mode.value, mode)
        mode_layout.addWidget(self.mode_combo)
        mode_group.setLayout(mode_layout)

        # board config
        board_group = QGroupBox("Board Configuration")
        board_layout = QGridLayout()
        board_layout.addWidget(QLabel("Size:"), 0, 0)
        self.size_combo = QComboBox()
        for size in [9, 13, 19]:
            self.size_combo.addItem(f"{size}×{size}", size)
        self.size_combo.setCurrentText("19×19")
        board_layout.addWidget(self.size_combo, 0, 1)

        board_layout.addWidget(QLabel("Handicap:"), 1, 0)
        self.handicap_spin = QSpinBox()
        self.handicap_spin.setRange(0, 9)
        board_layout.addWidget(self.handicap_spin, 1, 1)

        board_layout.addWidget(QLabel("Komi:"), 2, 0)
        self.komi_spin = QDoubleSpinBox()
        self.komi_spin.setRange(0.0, 10.0)
        self.komi_spin.setValue(6.5)
        self.komi_spin.setSingleStep(0.5)
        board_layout.addWidget(self.komi_spin, 2, 1)
        board_group.setLayout(board_layout)

        # AI Configuration
        # store as self for toggling
        self.ai_group = QGroupBox("AI Configuration")
        ai_layout = QGridLayout()

        ai_layout.addWidget(QLabel("Black AI:"), 0, 0)
        self.black_ai_combo = QComboBox()
        self.black_ai_combo.addItem("Human", "human")
        for model_type in AIModelType:
            self.black_ai_combo.addItem(model_type.value, model_type)
        ai_layout.addWidget(self.black_ai_combo, 0, 1)

        ai_layout.addWidget(QLabel("White AI:"), 1, 0)
        self.white_ai_combo = QComboBox()
        self.white_ai_combo.addItem("Human", "human")
        for model_type in AIModelType:
            self.white_ai_combo.addItem(model_type.value, model_type)
        ai_layout.addWidget(self.white_ai_combo, 1, 1)

        ai_layout.addWidget(QLabel("AI Strength:"), 2, 0)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(1, 10)
        self.strength_slider.setValue(5)
        self.strength_label = QLabel("5")
        self.strength_slider.valueChanged.connect(
            lambda v: self.strength_label.setText(str(v))
        )

        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_label)
        ai_layout.addLayout(strength_layout, 2, 1)
        self.ai_group.setLayout(ai_layout)

        # Analysis Configuration
        analysis_group = QGroupBox("Analysis Settings")
        analysis_layout = QGridLayout()
        self.realtime_analysis_cb = QCheckBox("Real-time Analysis")
        self.realtime_analysis_cb.setChecked(True)
        analysis_layout.addWidget(self.realtime_analysis_cb, 0, 0)
        self.show_variations_cb = QCheckBox("Show Variations")
        analysis_layout.addWidget(self.show_variations_cb, 0, 1)
        self.territory_analysis_cb = QCheckBox("Territory Analysis")
        analysis_layout.addWidget(self.territory_analysis_cb, 1, 0)
        analysis_group.setLayout(analysis_layout)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Add all groups to main layout
        layout.addWidget(mode_group)
        layout.addWidget(board_group)
        layout.addWidget(self.ai_group)
        layout.addWidget(analysis_group)
        layout.addWidget(buttons)
        self.setLayout(layout)

        # hide AI settings if human vs human
        self.mode_combo.currentIndexChanged.connect(self.toggle_ai_group)
        self.toggle_ai_group()  # initial state

    def toggle_ai_group(self):
        mode = self.mode_combo.currentData()
        self.ai_group.setVisible(mode != GameMode.HUMAN_VS_HUMAN)

    def get_settings(self):
        return {
            "mode": self.mode_combo.currentData(),
            "board_size": self.size_combo.currentData(),
            "handicap": self.handicap_spin.value(),
            "komi": self.komi_spin.value(),
            "black_ai_model": self.black_ai_combo.currentData(),
            "white_ai_model": self.white_ai_combo.currentData(),
            "ai_strength": self.strength_slider.value(),
            "realtime_analysis": self.realtime_analysis_cb.isChecked(),
            "show_variations": self.show_variations_cb.isChecked(),
            "territory_analysis": self.territory_analysis_cb.isChecked(),
        }
