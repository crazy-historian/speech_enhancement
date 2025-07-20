import sys
from PyQt6.QtWidgets import QApplication
from game.core_game import *
from guibouling import *


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BowlingConfigWindow()
    window.show()
    sys.exit(app.exec())
