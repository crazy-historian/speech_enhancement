import sys
from PyQt6.QtWidgets import QApplication
from guiboul import BowlingConfigWindow  # <-- это твой GUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BowlingConfigWindow()
    window.show()
    sys.exit(app.exec())