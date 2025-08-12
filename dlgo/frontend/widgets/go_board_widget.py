from typing import Optional, Dict

from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont
from PyQt6.QtCore import QPoint, Qt
from dlgo.gotypes import Player, Point

cols = "ABCDEFGHIJKLMNOPQRST"
EMPTY = None
BLACK = Player.black
WHITE = Player.white


class GoBoardWidget(QWidget):
    def __init__(self, board_size=19, parent=None):
        super().__init__(parent)
        self.board_size = board_size
        self.cell_size = 28
        self.margin = 40
        self.show_coordinates = True
        self.show_move_numbers = True
        self.show_analysis_overlay = True
        self.show_start_message = True

        canvas_size = (board_size - 1) * self.cell_size + 2 * self.margin
        self.setFixedSize(canvas_size, canvas_size)

        self.last_move = None
        self.candidate_moves = {}  # move -> evaluation
        self.variation_moves = []
        self.territory_analysis = {}

        self._setup_star_points()
        self._init_board_state()

        self.game_controller: Optional["GameController"] = None
        self.setStyleSheet("""
            background-color: #E6B873;
            border: 2px solid #8B4513;
            border-radius: 5px;
        """)

    def _setup_star_points(self):
        self.star_points = []
        if self.board_size == 19:
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
        elif self.board_size == 13:
            self.star_points = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif self.board_size == 9:
            self.star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]

    def _init_board_state(self):
        self.board_grid = [
            [EMPTY for _ in range(self.board_size)] for _ in range(self.board_size)
        ]

    def set_candidate_moves(self, moves_dict: Dict[str, float]):
        """Set candidate moves with their evaluations for visualization"""
        self.candidate_moves = moves_dict
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.show_coordinates:
            self._draw_coordinates(painter)

        self._draw_grid(painter)
        self._draw_star_points(painter)
        self._draw_stones(painter)
        if self.show_analysis_overlay:
            self._draw_analysis_overlay(painter)

        if self.last_move:
            self._draw_last_move_indicator(painter)

        painter.end()
        if self.show_start_message:
            painter = QPainter(self)

            # painter.setPen(QColor("white"))
            painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Select 'New Game' to begin"
            )

    def _draw_coordinates(self, painter):
        painter.setPen(QPen(QColor("#8B4513"), 1))
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)

        # Column labels (A-T)
        for i in range(self.board_size):
            x = i * self.cell_size + self.margin
            painter.drawText(x - 5, self.margin - 10, cols[i])
            painter.drawText(
                x - 5,
                (self.board_size - 1) * self.cell_size + self.margin + 20,
                cols[i],
            )

        # Row labels (1-19)
        for i in range(self.board_size):
            y = i * self.cell_size + self.margin
            row_num = self.board_size - i
            painter.drawText(self.margin - 25, y + 5, str(row_num))
            painter.drawText(
                (self.board_size - 1) * self.cell_size + self.margin + 10,
                y + 5,
                str(row_num),
            )

    def _draw_grid(self, painter):
        pen = QPen(QColor("#8B4513"), 1)
        painter.setPen(pen)

        for i in range(self.board_size):
            # Vertical lines
            x = i * self.cell_size + self.margin
            painter.drawLine(
                x, self.margin, x, (self.board_size - 1) *
                self.cell_size + self.margin
            )

            # Horizontal lines
            y = i * self.cell_size + self.margin
            painter.drawLine(
                self.margin, y, (self.board_size - 1) *
                self.cell_size + self.margin, y
            )

    def _draw_star_points(self, painter):
        brush = QBrush(QColor("#8B4513"))
        painter.setBrush(brush)
        for row, col in self.star_points:
            x, y = self.board_to_pixel(row, col)
            painter.drawEllipse(QPoint(x, y), 4, 4)

    def _draw_stones(self, painter):
        stone_radius = self.cell_size // 2 - 3

        for row in range(self.board_size):
            for col in range(self.board_size):
                stone = self.board_grid[row][col]
                if stone is not None:
                    x, y = self.board_to_pixel(row, col)

                    if stone == BLACK:
                        # Black stone with gradient
                        painter.setBrush(QBrush(QColor("#1a1a1a")))
                        painter.setPen(QPen(QColor("#000000"), 2))
                    else:  # WHITE
                        # White stone with subtle shadow
                        painter.setBrush(QBrush(QColor("#f5f5f5")))
                        painter.setPen(QPen(QColor("#333333"), 1))

                    painter.drawEllipse(
                        x - stone_radius,
                        y - stone_radius,
                        stone_radius * 2,
                        stone_radius * 2,
                    )

    def _draw_analysis_overlay(self, painter):
        """Draw AI analysis overlay showing move evaluations"""
        if not self.candidate_moves:
            return

        # Sort moves by evaluation
        sorted_moves = sorted(
            self.candidate_moves.items(), key=lambda x: x[1], reverse=True
        )

        for i, (move_str, evaluation) in enumerate(
            sorted_moves[:10]
        ):  # Show top 10 moves
            if move_str == "pass":
                continue

            try:
                # Parse move string (e.g., "D4" -> row=4, col=4)
                col_char = move_str[0]
                row_num = int(move_str[1:])

                col = cols.index(col_char)
                row = self.board_size - row_num

                if 0 <= row < self.board_size and 0 <= col < self.board_size:
                    x, y = self.board_to_pixel(row, col)

                    # Color based on evaluation (green = good, red = bad)
                    eval_normalized = max(
                        0, min(1, (evaluation + 1) / 2)
                    )  # Normalize to [0,1]
                    color = QColor()
                    if eval_normalized > 0.5:
                        # Green tones for good moves
                        color.setRgb(
                            int(255 * (1 - eval_normalized)),
                            255,
                            int(255 * (1 - eval_normalized)),
                            128,
                        )
                    else:
                        # Red tones for bad moves
                        color.setRgb(
                            255,
                            int(255 * eval_normalized),
                            int(255 * eval_normalized),
                            128,
                        )

                    # Draw evaluation circle
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(QColor("#000000"), 1))
                    radius = 8 + (3 if i < 3 else 0)  # Larger for top moves
                    painter.drawEllipse(
                        x - radius, y - radius, radius * 2, radius * 2)

                    # Draw evaluation text
                    painter.setPen(QPen(QColor("#000000")))
                    font = QFont("Arial", 8, QFont.Weight.Bold)
                    painter.setFont(font)
                    text = f"{evaluation:.2f}"
                    painter.drawText(x - 15, y - radius - 5, text)

            except (ValueError, IndexError):
                continue

    def _draw_last_move_indicator(self, painter):
        """Draw indicator for the last move played"""
        if not self.last_move:
            return

        row, col = self.last_move
        x, y = self.board_to_pixel(row, col)

        # Draw red circle around last move
        painter.setPen(QPen(QColor("#FF0000"), 3))
        painter.setBrush(QBrush())  # No fill
        painter.drawEllipse(x - 15, y - 15, 30, 30)

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
            row, col = self.pixel_to_board(
                event.position().x(), event.position().y())
            if row < 0 or col < 0 or row >= self.board_size or col >= self.board_size:
                return

            # prevent move if there's already a stone here
            if self.board_grid[row][col] is not None:
                return

            self.game_controller.human_move_attempt(row + 1, col + 1)

    def update_board(self, game_state):
        if game_state is None:
            return

        board = getattr(game_state, "board", None)
        if board is None:
            return

        for r in range(self.board_size):
            for c in range(self.board_size):
                # gotypes.Point uses 1-based coordinates
                pt = Point(row=r + 1, col=c + 1)
                stone = None
                try:
                    stone = board.get(pt)
                except Exception:
                    try:
                        stone = board._grid[r][c]
                    except Exception:
                        stone = None
                self.board_grid[r][c] = stone

        last_move = getattr(game_state, "last_move", None)
        if last_move is not None:
            try:
                self.last_move = (last_move.row - 1, last_move.col - 1)
            except Exception:
                self.last_move = None
        else:
            self.last_move = None

        self.update()
