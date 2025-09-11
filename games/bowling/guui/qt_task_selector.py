from __future__ import annotations
import sys, time
from typing import Optional
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from .profiles import get_tasks

class TaskPickerDialog(QDialog):
    def __init__(self, profile_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Выбор задания — {profile_name}")
        self.resize(520, 520)
        self.selected_task: Optional[dict] = None
        self._tasks_cache = get_tasks(profile_name)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Профиль: {profile_name}"))
        layout.addWidget(QLabel("Двойной клик — выбрать"))

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._accept_current)
        layout.addWidget(self.list_widget)
        for t in self._tasks_cache:
            self.list_widget.addItem(t.get("name", "(без названия)"))
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

        btns = QHBoxLayout()
        ok = QPushButton("Выбрать"); cancel = QPushButton("Отмена")
        ok.clicked.connect(self._accept_current); cancel.clicked.connect(self.reject)
        btns.addWidget(ok); btns.addWidget(cancel)
        layout.addLayout(btns)

    def _accept_current(self):
        it = self.list_widget.currentItem()
        if not it: return
        name = it.text()
        for t in self._tasks_cache:
            if t.get("name") == name:
                self.selected_task = t
                break
        self.accept()

def select_task_for_profile(profile_name: str):
    """
    Модальный выбор задания без dlg.exec(): крутим события вручную.
    Возвращает dict | None.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    dlg = TaskPickerDialog(profile_name)
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

    return dlg.selected_task
