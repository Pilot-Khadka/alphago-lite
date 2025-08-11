from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QGroupBox,
    QProgressBar,
    QGridLayout,
    QHeaderView,
    QSizePolicy,
    QScrollArea,
)


class AnalysisWidget(QWidget):
    """Displays game analysis and AI insights"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # scrollable iner content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner_widget = QWidget()
        inner_layout = QVBoxLayout()

        # position evaluation
        eval_group = QGroupBox("Position Evaluation")
        eval_layout = QVBoxLayout()
        self.value_label = QLabel("Position Value: N/A")
        self.value_label.setFont(QFont("Consolas", 12))
        eval_layout.addWidget(self.value_label)

        self.winprob_bar = QProgressBar()
        self.winprob_bar.setRange(0, 100)
        self.winprob_bar.setValue(50)
        self.winprob_bar.setFormat("Black: %p%")
        eval_layout.addWidget(self.winprob_bar)
        eval_group.setLayout(eval_layout)

        moves_group = QGroupBox("Move Analysis")
        moves_layout = QVBoxLayout()

        self.moves_table = QTableWidget()
        self.moves_table.setColumnCount(4)
        self.moves_table.setHorizontalHeaderLabels(
            ["Move", "Win Rate", "Visits", "Policy"]
        )
        self.moves_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        # limit height so stats are always visible
        self.moves_table.setMaximumHeight(180)
        self.moves_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.moves_table.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        moves_layout.addWidget(self.moves_table)
        moves_group.setLayout(moves_layout)

        stats_group = QGroupBox("Computational Statistics")
        stats_layout = QGridLayout()
        self.nodes_label = QLabel("Nodes/sec: 0")
        self.time_label = QLabel("Search time: 0.0s")
        self.depth_label = QLabel("Max depth: 0")
        self.gpu_util_label = QLabel("GPU Util: 0%")

        stats_layout.addWidget(self.nodes_label, 0, 0)
        stats_layout.addWidget(self.time_label, 0, 1)
        stats_layout.addWidget(self.depth_label, 1, 0)
        stats_layout.addWidget(self.gpu_util_label, 1, 1)
        stats_group.setLayout(stats_layout)

        # add widgets to inner layout
        inner_layout.addWidget(eval_group)
        inner_layout.addWidget(moves_group)
        inner_layout.addWidget(stats_group)
        inner_layout.addStretch()

        inner_widget.setLayout(inner_layout)
        scroll.setWidget(inner_widget)

        # add scroll area to main layout
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
