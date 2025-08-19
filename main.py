import sys
from PyQt6.QtWidgets import QApplication
from dlgo.frontend.widgets.main_window_widget import GoAIMainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Go AI")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Go AI")
    window = GoAIMainWindow(19)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
