import time
import queue
import threading
from typing import Callable
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QDialog,
    QSplitter,
    QTabWidget,
    QFrame,
)
from PyQt6.QtCore import QTimer
from dlgo.gotypes import Player
from dlgo import gotypes, goboard
from dlgo.goboard import Board, GameState
from dlgo.configs.types import GameAnalysis, AIModelType, GameMode
from dlgo.frontend.widgets.go_board_widget import GoBoardWidget
from dlgo.frontend.widgets.analysis_widget import AnalysisWidget
from dlgo.frontend.widgets.model_config_widget import ModelConfigWidget
from dlgo.frontend.widgets.log_widget import LogWidget
from dlgo.frontend.widgets.game_setup_dialogue_widget import GameSetupDialog

cols = "ABCDEFGHIJKLMNOPQRST"
EMPTY = None
BLACK = Player.black
WHITE = Player.white


class AIPlayer:
    def __init__(
        self, player_color: Player, model_type: AIModelType = None, name: str = ""
    ):
        self.player_color = player_color
        self.model_type = model_type or AIModelType.MCTS_PURE
        self.name = name or f"AI ({self.model_type.value})"

        # AI-specific attributes
        self.model = None
        self.mcts_simulations = 1600
        self.temperature = 1.0
        self.analysis_callback = None

        # Move computation
        self.pending_move = None
        self.computing = False
        self.computation_thread = None

    def start_analysis(self, position, callback):
        """Start continuous analysis of position"""
        self.analysis_callback = callback
        # TODO: Implement continuous analysis

    def get_move(self, game_state):
        if self.pending_move:
            move = self.pending_move
            self.pending_move = None
            self.computing = False
            return move
        elif not self.computing:
            self._start_move_computation(game_state)
        return None

    def _start_move_computation(self, game_state):
        """Start computing move in background thread"""
        self.computing = True

        # TODO: Implement actual AI move computation
        # For now, simulate with timer
        def simulate_computation():
            time.sleep(1.0)  # Simulate computation time
            # Generate dummy move (pass for now)
            self.pending_move = goboard.Move.pass_turn()

        self.computation_thread = threading.Thread(target=simulate_computation)
        self.computation_thread.start()


class HumanPlayer:
    """Human player implementation"""

    def __init__(self, player_color: Player, name: str = ""):
        self.player_color = player_color
        self.name = name or f"{player_color.name.title()} Player"
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


class GameController:
    def __init__(self, board_size: int = 19):
        self.board_size = board_size
        self.board = Board(size=board_size)
        self.game_state = GameState.new_game(board_size=board_size)
        self.players = {Player.black: None, Player.white: None}

        # Analysis and logging
        self.move_history = []
        self.analysis_history = []
        self.move_callbacks = []
        self.game_over_callbacks = []
        self.analysis_callbacks = []

        # AI computation thread
        self.ai_thread = None
        self.computation_queue = queue.Queue()

        # Timer for updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._process_updates)
        self.update_timer.start(50)  # 20 FPS updates

    def set_players(self, black_player, white_player):
        self.players = {Player.black: black_player, Player.white: white_player}

    def start_analysis_mode(self, position):
        """Start continuous analysis of current position"""
        current_player = self.get_current_player()
        if hasattr(current_player, "start_analysis"):
            current_player.start_analysis(position, self._on_analysis_update)

    def _on_analysis_update(self, analysis: GameAnalysis):
        """Handle analysis updates from AI"""
        for callback in self.analysis_callbacks:
            callback(analysis)

    def add_analysis_callback(self, callback: Callable):
        self.analysis_callbacks.append(callback)

    def _process_updates(self):
        """Process queued updates and AI computations"""
        if not hasattr(self, "players"):
            return

        try:
            while True:
                analysis = self.computation_queue.get_nowait()
                self._on_analysis_update(analysis)
        except queue.Empty:
            pass

        if not self.game_state.is_over():
            current_color = self.game_state.next_player  # player enum
            current_player = self.players[current_color]
            if current_player:
                move = current_player.get_move(self.game_state)
                if move:
                    self._apply_move(move)

    def start_new_game(self, game_state):
        self.game_state = game_state
        self.current_player_color = BLACK
        self.game_over = False

    def human_move_attempt(self, row: int, col: int) -> bool:
        if self.game_state.is_over():
            return False

        current_color = self.game_state.next_player
        current_player = self.players[current_color]

        if not isinstance(current_player, HumanPlayer):
            return False  # ignore clicks if it's not the human's turn

        point = gotypes.Point(row=row, col=col)
        move = goboard.Move.play(point)

        if self.game_state.is_valid_move(move):
            current_player.set_move(move)
            return True
        return False

    def _apply_move(self, move: goboard.Move):
        self.game_state = self.game_state.apply_move(move)

        for callback in self.move_callbacks:
            callback(self.game_state.next_player, move, self.game_state)

        if self.game_state.is_over():
            self.game_over = True
            for callback in self.game_over_callbacks:
                callback()

    def add_move_callback(self, callback: Callable):
        self.move_callbacks.append(callback)

    def add_game_over_callback(self, callback: Callable):
        self.game_over_callbacks.append(callback)


class GoAIMainWindow(QMainWindow):
    def __init__(self, board_size=19):
        super().__init__()
        self.board_size = board_size
        self.game_controller = GameController(board_size)

        self.setWindowTitle("Go AI - AlphaGo Style Implementation")
        self.setMinimumSize(1400, 900)
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        self._apply_professional_styling()

    def _setup_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel - Board and controls
        left_panel = QVBoxLayout()

        # Board widget
        self.board_widget = GoBoardWidget(self.board_size)
        self.board_widget.game_controller = self.game_controller
        left_panel.addWidget(self.board_widget)

        # Game controls
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_layout = QHBoxLayout()

        self.pass_btn = QPushButton("Pass")
        self.resign_btn = QPushButton("Resign")
        self.analyze_btn = QPushButton("Analyze Position")
        self.undo_btn = QPushButton("Undo")

        controls_layout.addWidget(self.pass_btn)
        controls_layout.addWidget(self.resign_btn)
        controls_layout.addWidget(self.analyze_btn)
        controls_layout.addWidget(self.undo_btn)
        controls_layout.addStretch()

        controls_frame.setLayout(controls_layout)
        left_panel.addWidget(controls_frame)

        # Right panel - Analysis and configuration
        right_panel = QTabWidget()

        # Analysis tab
        self.analysis_widget = AnalysisWidget()
        right_panel.addTab(self.analysis_widget, "Analysis")

        # Model configuration tab
        self.model_config_widget = ModelConfigWidget()
        right_panel.addTab(self.model_config_widget, "AI Models")

        # Logging tab
        self.log_widget = LogWidget()
        right_panel.addTab(self.log_widget, "Game Log")

        # Splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_panel)

        # Set initial sizes (board larger than panels)
        splitter.setSizes([800, 600])

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New Game", self.new_game)
        file_menu.addAction("Load Game", self.load_game)
        file_menu.addAction("Save Game", self.save_game)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # AI menu
        ai_menu = menubar.addMenu("AI")
        ai_menu.addAction("Load Model", self.load_model)
        ai_menu.addAction("Model Comparison", self.compare_models)
        ai_menu.addSeparator()
        ai_menu.addAction("Start Analysis", self.start_continuous_analysis)
        ai_menu.addAction("Stop Analysis", self.stop_analysis)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction("Position Editor", self.open_position_editor)
        tools_menu.addAction("SGF Viewer", self.open_sgf_viewer)
        tools_menu.addAction("Statistics", self.view_statistics)

    def _connect_signals(self):
        # Connect game controller
        self.game_controller.add_move_callback(self._on_move_made)
        self.game_controller.add_analysis_callback(self._on_analysis_update)

        # Connect UI elements
        self.analyze_btn.clicked.connect(self.start_position_analysis)

    def _apply_professional_styling(self):
        """Apply professional dark theme styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #3c3c3c;
            }
            QTabBar::tab {
                background-color: #555;
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #3c3c3c;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)

    def _on_move_made(self, player_color, move, game_state):
        """Handle move made callback"""
        move_str = self._format_move(move)
        player_name = "Black" if player_color == BLACK else "White"

        # Log the move
        self.log_widget.log_move(
            player_name, move_str, 0.0, 0.0
        )  # TODO: Add real evaluation and timing

        # Update board display
        if hasattr(self.board_widget, "update_board"):
            self.board_widget.update_board(game_state)

    def _on_analysis_update(self, analysis: GameAnalysis):
        """Handle analysis update"""
        self.analysis_widget.update_analysis(analysis)

        # Update board with candidate moves
        if hasattr(analysis, "policy_distribution"):
            self.board_widget.set_candidate_moves(analysis.policy_distribution)

    def start_position_analysis(self):
        """Start analyzing current position"""
        self.analyze_btn.setText("Analyzing...")
        self.analyze_btn.setEnabled(False)

        # TODO: Implement actual position analysis
        self.log_widget.log_analysis("Starting position analysis...")

        # Simulate analysis completion after delay
        QTimer.singleShot(2000, self._analysis_complete)

    def _analysis_complete(self):
        """Handle analysis completion"""
        self.analyze_btn.setText("Analyze Position")
        self.analyze_btn.setEnabled(True)
        self.log_widget.log_analysis("Position analysis complete")

    def _format_move(self, move):
        """Format a move for display"""
        if move.is_pass:
            return "Pass"
        elif move.is_resign:
            return "Resign"
        else:
            col_letter = cols[move.point.col - 1]
            return f"{col_letter}{move.point.row}"

    # Menu action implementations
    def new_game(self):
        dialog = GameSetupDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            self._start_new_game(settings)

    def load_game(self):
        """Load a game from SGF file"""
        self.log_widget.log_analysis("Load game functionality to be implemented")

    def save_game(self):
        """Save current game to SGF file"""
        self.log_widget.log_analysis("Save game functionality to be implemented")

    def load_model(self):
        """Load AI model"""
        self.log_widget.log_analysis("Model loading functionality to be implemented")

    def compare_models(self):
        """Open model comparison interface"""
        self.log_widget.log_analysis("Model comparison functionality to be implemented")

    def start_continuous_analysis(self):
        """Start continuous position analysis"""
        self.game_controller.start_analysis_mode(self.game_controller.game_state)
        self.log_widget.log_analysis("Started continuous analysis")

    def stop_analysis(self):
        """Stop continuous analysis"""
        self.log_widget.log_analysis("Stopped continuous analysis")

    def open_position_editor(self):
        """Open position editor"""
        self.log_widget.log_analysis("Position editor functionality to be implemented")

    def open_sgf_viewer(self):
        """Open SGF viewer"""
        self.log_widget.log_analysis("SGF viewer functionality to be implemented")

    def view_statistics(self):
        """View game and model statistics"""
        dialog = StatisticsDialog(self)
        dialog.exec()

    def _start_new_game(self, settings):
        mode = settings["mode"]
        board_size = settings["board_size"]

        # create players and reset game state
        black_player, white_player = self._create_players_from_settings(settings)
        self.game_controller.set_players(black_player, white_player)

        # create new GameState
        new_game_state = GameState.new_game(board_size=board_size)
        self.game_controller.start_new_game(new_game_state)

        if hasattr(self.board_widget, "update_board"):
            self.board_widget.update_board(new_game_state)

        self.log_widget.log_analysis(
            f"""Started new game: {mode.value} on {board_size}x{board_size} board"""
        )

    def _create_players_from_settings(self, settings):
        # TODO: Implement player creation with AI model selection
        mode = settings["mode"]

        if mode == GameMode.HUMAN_VS_AI:
            return HumanPlayer(BLACK, "Human"), AIPlayer(
                WHITE, settings.get("ai_model")
            )
        elif mode == GameMode.AI_VS_AI:
            return AIPlayer(BLACK, settings.get("black_ai_model")), AIPlayer(
                WHITE, settings.get("white_ai_model")
            )
        else:
            return HumanPlayer(BLACK, "Black"), HumanPlayer(WHITE, "White")
