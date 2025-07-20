import sys
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable, Tuple
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QDialog,
    QDialogButtonBox,
)
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QAction
from PyQt6.QtCore import Qt, QPoint, QTimer
from dlgo import gotypes
from dlgo import goboard
from dlgo.gotypes import Player

cols = "ABCDEFGHIJKLMNOPQRST"
stone_to_char = {None: ".", Player.black: "X", Player.white: "O"}
EMPTY = None
BLACK = Player.black
WHITE = Player.white


class GameMode(Enum):
    HUMAN_VS_HUMAN = "Human vs Human"
    HUMAN_VS_BOT = "Human vs Bot"
    BOT_VS_BOT = "Bot vs Bot"
    HUMAN_VS_SERVER = "Human vs Server"
    BOT_VS_SERVER = "Bot vs Server"


class PlayerType(Enum):
    HUMAN = "human"
    LOCAL_BOT = "local_bot"
    SERVER_BOT = "server_bot"


def print_move(player, move):
    if move.is_pass:
        move_str = "passes"
    elif move.is_resign:
        move_str = "resigns"
    else:
        move_str = "%s%d" % (cols[move.point.col - 1], move.point.row)
        print("%s %s" % (player, move_str))


class GamePlayer(ABC):
    def __init__(self, player_color: Player, name: str = ""):
        self.player_color = player_color
        self.name = name or f"{player_color.name.title()} Player"

    @abstractmethod
    def get_move(self, game_state):
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the player is ready to make a move"""
        pass


class HumanPlayer(GamePlayer):
    def __init__(self, player_color: Player, name: str = ""):
        super().__init__(player_color, name)
        self.pending_move = None
        self.move_ready = False

    def get_move(self, game_state):
        if self.move_ready and self.pending_move:
            move = self.pending_move
            self.pending_move = None
            self.move_ready = False
            return move
        return None

    def set_move(self, move):
        self.pending_move = move
        self.move_ready = True

    def is_ready(self) -> bool:
        return self.move_ready


class LocalBotPlayer(GamePlayer):
    def __init__(self, player_color: Player, bot_strategy=None, name: str = ""):
        super().__init__(player_color, name or "Local Bot")
        self.bot_strategy = bot_strategy

    def get_move(self, game_state):
        pass

    def is_ready(self) -> bool:
        return True


class ServerBotPlayer(GamePlayer):
    def __init__(self, player_color: Player, server_connection=None, name: str = ""):
        super().__init__(player_color, name or "Server Bot")
        self.server_connection = server_connection
        self.pending_move = None
        self.waiting_for_server = False

    def get_move(self, game_state):
        if self.pending_move:
            move = self.pending_move
            self.pending_move = None
            self.waiting_for_server = False
            return move
        elif not self.waiting_for_server:
            self._request_move_from_server(game_state)
            self.waiting_for_server = True
        return None

    def _request_move_from_server(self, game_state):
        # TODO
        pass

    def receive_move_from_server(self, move):
        self.pending_move = move

    def is_ready(self) -> bool:
        return self.pending_move is not None


class GameController:
    """Manages game flow and player interactions"""

    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.game_state = None  # Will hold the actual game state
        self.black_player: Optional[GamePlayer] = None
        self.white_player: Optional[GamePlayer] = None
        self.current_player_color = BLACK
        self.game_over = False
        self.move_callbacks = []
        self.game_over_callbacks = []

        # Timer for bot moves and animations
        self.move_timer = QTimer()
        self.move_timer.timeout.connect(self._process_moves)
        self.move_timer.start(100)  # Check every 100ms

    def set_players(self, black_player: GamePlayer, white_player: GamePlayer):
        self.black_player = black_player
        self.white_player = white_player

    def get_current_player(self) -> Optional[GamePlayer]:
        if self.current_player_color == BLACK:
            return self.black_player
        return self.white_player

    def start_new_game(self, game_state):
        self.game_state = game_state
        self.current_player_color = BLACK
        self.game_over = False

    def human_move_attempt(self, row: int, col: int) -> bool:
        """Handle human player move attempt"""
        current_player = self.get_current_player()
        if not isinstance(current_player, HumanPlayer) or self.game_over:
            return False

        try:
            point = gotypes.Point(row=row, col=col)
            move = goboard.Move.play(point)

            if self._is_valid_move(move):
                current_player.set_move(move)
                return True
        except:
            pass
        return False

    def _is_valid_move(self, move: goboard.Move) -> bool:
        return True

    def _process_moves(self):
        """Process pending moves from current player"""
        if self.game_over:
            return

        current_player = self.get_current_player()
        if current_player and current_player.is_ready():
            move = current_player.get_move(self.game_state)
            if move:
                self._apply_move(move)

    def _apply_move(self, move: goboard.Move):
        """Apply a move to the game state"""
        # self.game_state = self.game_state.apply_move(move)

        # Notify UI components
        for callback in self.move_callbacks:
            callback(self.current_player_color, move, self.game_state)

        # Switch players
        if self.current_player_color == BLACK:
            self.current_player_color = WHITE
        else:
            self.current_player_color = BLACK

        # Check for game over
        if self._check_game_over():
            self._handle_game_over()

    def _check_game_over(self) -> bool:
        # Implement game over logic
        return False

    def _handle_game_over(self):
        self.game_over = True
        for callback in self.game_over_callbacks:
            callback()

    def add_move_callback(self, callback: Callable):
        self.move_callbacks.append(callback)

    def add_game_over_callback(self, callback: Callable):
        self.game_over_callbacks.append(callback)


class GoBoardWidget(QWidget):
    def __init__(self, board_size=19, parent=None):
        super().__init__(parent)
        self.board_size = board_size
        self.cell_size = 25
        self.margin = 30

        canvas_size = (board_size - 1) * self.cell_size + 2 * self.margin
        self.setFixedSize(canvas_size, canvas_size)

        self.star_points = []
        if board_size == 19:
            self.star_points = [
                (3, 3),
                (3, 9),
                (3, 15),
                (9, 3),
                (9, 9),
                (9, 15),
                (15, 3),
                (15, 9),
                (15, 15),
            ]

        self.board_grid = [
            [EMPTY for _ in range(board_size)] for _ in range(board_size)
        ]

        self.game_controller: Optional[GameController] = None
        self.setStyleSheet("background-color: #dcb35c")

    def set_game_controller(self, controller: GameController):
        self.game_controller = controller

    def board_to_pixel(self, row, col):
        x = col * self.cell_size + self.margin
        y = row * self.cell_size + self.margin
        return x, y

    def pixel_to_board(self, x, y):
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        return row, col

    def mousePressEvent(self, event):
        if self.game_controller and event.button() == Qt.MouseButton.LeftButton:
            row, col = self.pixel_to_board(event.x(), event.y())
            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                self.game_controller.human_move_attempt(row + 1, col + 1)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(QColor("black"), 1)
        painter.setPen(pen)

        # Draw grid lines
        for i in range(self.board_size):
            x = i * self.cell_size + self.margin
            painter.drawLine(
                x, self.margin, x, (self.board_size - 1) * self.cell_size + self.margin
            )
            y = i * self.cell_size + self.margin
            painter.drawLine(
                self.margin, y, (self.board_size - 1) * self.cell_size + self.margin, y
            )

        # Draw star points
        brush = QBrush(QColor("black"))
        painter.setBrush(brush)
        for row, col in self.star_points:
            x, y = self.board_to_pixel(row, col)
            painter.drawEllipse(QPoint(x, y), 3, 3)

        # Draw stones
        stone_radius = self.cell_size // 2 - 2
        for row in range(self.board_size):
            for col in range(self.board_size):
                stone = self.board_grid[row][col]
                if stone is not None:
                    x, y = self.board_to_pixel(row, col)

                    if stone == BLACK:
                        painter.setBrush(QBrush(QColor("black")))
                        painter.setPen(QPen(QColor("black"), 1))
                        painter.drawEllipse(
                            x - stone_radius,
                            y - stone_radius,
                            stone_radius * 2,
                            stone_radius * 2,
                        )
                    elif stone == WHITE:
                        painter.setBrush(QBrush(QColor("white")))
                        painter.setPen(QPen(QColor("black"), 2))
                        painter.drawEllipse(
                            x - stone_radius,
                            y - stone_radius,
                            stone_radius * 2,
                            stone_radius * 2,
                        )

    def update_board(self, board):
        """Update the visual board display"""
        for row in range(1, board.size + 1):
            for col in range(1, board.size + 1):
                stone = board.get(gotypes.Point(row=row, col=col))
                self.board_grid[row - 1][col - 1] = stone
        self.update()


class GameSetupDialog(QDialog):
    """Dialog for setting up new games"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Game Setup")
        self.setModal(True)

        layout = QVBoxLayout()

        # Game mode selection
        layout.addWidget(QLabel("Game Mode:"))
        self.mode_combo = QComboBox()
        for mode in GameMode:
            self.mode_combo.addItem(mode.value, mode)
        layout.addWidget(self.mode_combo)

        # Board size selection
        layout.addWidget(QLabel("Board Size:"))
        self.size_combo = QComboBox()
        for size in [9, 13, 19]:
            self.size_combo.addItem(f"{size}x{size}", size)
        self.size_combo.setCurrentText("19x19")
        layout.addWidget(self.size_combo)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_settings(self):
        return {
            "mode": self.mode_combo.currentData(),
            "board_size": self.size_combo.currentData(),
        }


class GameStatusWidget(QWidget):
    """Widget to display game status and controls"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()

        self.status_label = QLabel("Ready to start")
        self.pass_button = QPushButton("Pass")
        self.resign_button = QPushButton("Resign")

        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.pass_button)
        layout.addWidget(self.resign_button)

        self.setLayout(layout)

        # Initially disable buttons
        self.pass_button.setEnabled(False)
        self.resign_button.setEnabled(False)

    def update_status(self, message: str):
        self.status_label.setText(message)

    def set_controls_enabled(self, enabled: bool):
        self.pass_button.setEnabled(enabled)
        self.resign_button.setEnabled(enabled)


class GoBoardApplication(QMainWindow):
    """Main application window"""

    def __init__(self, board_size=19):
        super().__init__()
        self.board_size = board_size
        self.game_controller = GameController(board_size)

        self.setWindowTitle("Go Game")
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

    def _setup_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Board widget
        self.board_widget = GoBoardWidget(self.board_size)
        self.board_widget.set_game_controller(self.game_controller)
        layout.addWidget(self.board_widget)

        # Status widget
        self.status_widget = GameStatusWidget()
        layout.addWidget(self.status_widget)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Adjust window size
        self.setFixedSize(
            (self.board_size - 1) * 25 + 2 * 30 + 20,
            (self.board_size - 1) * 25 + 2 * 30 + 100,
        )

    def _setup_menu(self):
        menubar = self.menuBar()

        # Game menu
        game_menu = menubar.addMenu("Game")

        new_game_action = QAction("New Game", self)
        new_game_action.triggered.connect(self.new_game)
        game_menu.addAction(new_game_action)

        game_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        game_menu.addAction(exit_action)

    def _connect_signals(self):
        # Connect game controller callbacks
        self.game_controller.add_move_callback(self._on_move_made)
        self.game_controller.add_game_over_callback(self._on_game_over)

        # Connect status widget buttons
        self.status_widget.pass_button.clicked.connect(self._on_pass_clicked)
        self.status_widget.resign_button.clicked.connect(self._on_resign_clicked)

    def new_game(self):
        dialog = GameSetupDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self._start_new_game(settings["mode"], settings["board_size"])

    def _start_new_game(self, mode: GameMode, board_size: int):
        # Create players based on mode
        black_player, white_player = self._create_players(mode)

        # Set up game controller
        self.game_controller.set_players(black_player, white_player)

        # Initialize game state (you'll need to implement this with your game logic)
        # game_state = YourGameState(board_size)
        # self.game_controller.start_new_game(game_state)

        # Update UI
        self.status_widget.update_status(
            f"{mode.value} - {black_player.name} vs {white_player.name}"
        )
        self.status_widget.set_controls_enabled(True)

    def _create_players(self, mode: GameMode) -> Tuple[GamePlayer, GamePlayer]:
        """Factory method to create players based on game mode"""
        if mode == GameMode.HUMAN_VS_HUMAN:
            return HumanPlayer(BLACK, "Black"), HumanPlayer(WHITE, "White")
        elif mode == GameMode.HUMAN_VS_BOT:
            return HumanPlayer(BLACK, "Human"), LocalBotPlayer(WHITE)
        elif mode == GameMode.BOT_VS_BOT:
            return LocalBotPlayer(BLACK), LocalBotPlayer(WHITE)
        elif mode == GameMode.HUMAN_VS_SERVER:
            return HumanPlayer(BLACK, "Human"), ServerBotPlayer(WHITE)
        elif mode == GameMode.BOT_VS_SERVER:
            return LocalBotPlayer(BLACK), ServerBotPlayer(WHITE)
        else:
            return HumanPlayer(BLACK, "Black"), HumanPlayer(WHITE, "White")

    def _on_move_made(self, player_color, move, game_state):
        """Handle when a move is made"""
        self.board_widget.update_board_display(game_state)

        current_player = self.game_controller.get_current_player()
        if current_player:
            self.status_widget.update_status(f"{current_player.name}'s turn")

    def _on_game_over(self):
        """Handle game over"""
        self.status_widget.update_status("Game Over")
        self.status_widget.set_controls_enabled(False)

    def _on_pass_clicked(self):
        """Handle pass button click"""
        current_player = self.game_controller.get_current_player()
        if isinstance(current_player, HumanPlayer):
            current_player.set_move(goboard.Move.pass_turn())

    def _on_resign_clicked(self):
        """Handle resign button click"""
        current_player = self.game_controller.get_current_player()
        if isinstance(current_player, HumanPlayer):
            current_player.set_move(goboard.Move.resign())


def main():
    app = QApplication(sys.argv)
    window = GoBoardApplication(19)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
