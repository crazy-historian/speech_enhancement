from __future__ import annotations
import sys, time
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from .profiles import load_profiles

class ProfilePickerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор профиля — Bowling")
        self.resize(420, 420)
        self.selected_name = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Двойной клик — выбрать"))

        self.list_widget = QListWidget()
        data = load_profiles()
        for p in data.get("profiles", []):
            self.list_widget.addItem(p.get("name"))
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)
        self.list_widget.itemDoubleClicked.connect(self._accept_current)
        layout.addWidget(self.list_widget)

        btns = QHBoxLayout()
        ok = QPushButton("Выбрать"); cancel = QPushButton("Отмена")
        ok.clicked.connect(self._accept_current); cancel.clicked.connect(self.reject)
        btns.addWidget(ok); btns.addWidget(cancel)
        layout.addLayout(btns)

    def _accept_current(self):
        it = self.list_widget.currentItem()
        if not it: return
        self.selected_name = it.text()
        self.accept()

def select_profile_name():
    """
    Модальный выбор профиля без dlg.exec(): крутим событийный цикл вручную.
    Возвращает str | None.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    dlg = ProfilePickerDialog()
    dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.show(); app.processEvents()

    done = {"d": False}
    def _ok(): done["d"] = True
    def _no(): done["d"] = True
    dlg.accepted.connect(_ok); dlg.rejected.connect(_no)

    while dlg.isVisible() and not done["d"]:
        app.processEvents()
        time.sleep(0.01)

    try:
        if dlg.isVisible(): dlg.close()
    except Exception:
        pass

    return dlg.selected_name
