import sys
from PyQt6.QtWidgets import QApplication
from .profile_select import ProfileSelectWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = ProfileSelectWindow()
    w.show()
    sys.exit(app.exec())