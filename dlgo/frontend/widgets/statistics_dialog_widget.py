from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QDialog,
    QTableWidget,
    QTabWidget,
)


class StatisticsDialog(QDialog):
    """Dialog for viewing game and model statistics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Statistics ")
        self.setModal(True)
        self.setMinimumSize(800, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        tabs = QTabWidget()

        game_stats_widget = QWidget()
        game_stats_layout = QVBoxLayout()

        self.games_table = QTableWidget()
        self.games_table.setColumnCount(6)
        self.games_table.setHorizontalHeaderLabels(
            ["Date", "Black Player", "White Player", "Result", "Moves", "Duration"]
        )
        game_stats_layout.addWidget(self.games_table)

        game_stats_widget.setLayout(game_stats_layout)
        tabs.addTab(game_stats_widget, "Games")

        # model Performance
        model_stats_widget = QWidget()
        model_stats_layout = QVBoxLayout()

        self.models_table = QTableWidget()
        self.models_table.setColumnCount(5)
        self.models_table.setHorizontalHeaderLabels(
            ["Model", "Win Rate", "Avg Move Time", "Games Played", "ELO Rating"]
        )
        model_stats_layout.addWidget(self.models_table)

        model_stats_widget.setLayout(model_stats_layout)
        tabs.addTab(model_stats_widget, "Model Performance")

        # system Performance
        system_stats_widget = QWidget()
        system_stats_layout = QVBoxLayout()

        # TODO: add system performance graphs and metrics
        system_stats_layout.addWidget(
            QLabel("System performance metrics will be displayed here")
        )

        system_stats_widget.setLayout(system_stats_layout)
        tabs.addTab(system_stats_widget, "System Performance")

        layout.addWidget(tabs)

        # close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)
