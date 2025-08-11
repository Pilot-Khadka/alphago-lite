from datetime import datetime
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QTextEdit


class LogWidget(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        self.setMaximumHeight(200)

    def log_move(self, player: str, move: str, evaluation: float, time_taken: float):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"""[{timestamp}] {player}: {move} (eval: {evaluation:.3f}, time: {
            time_taken:.2f}s)"""
        self.append(log_entry)

    def log_analysis(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.append(f"[{timestamp}] ANALYSIS: {message}")

    def log_error(self, error: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.append(f"[{timestamp}] ERROR: {error}")
