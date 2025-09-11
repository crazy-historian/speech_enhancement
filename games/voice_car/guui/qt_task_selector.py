from __future__ import annotations
from typing import Optional
import sys

__all__ = ["select_task_for_profile", "TaskPickerDialog"]

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QListWidget, QPushButton,
    QHBoxLayout, QLabel
)
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
        btn_ok = QPushButton("Выбрать")
        btn_cancel = QPushButton("Отмена")
        btn_ok.clicked.connect(self._accept_current)
        btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_ok); btns.addWidget(btn_cancel)
        layout.addLayout(btns)

    def _accept_current(self):
        it = self.list_widget.currentItem()
        if not it:
            return
        name = it.text()
        for t in self._tasks_cache:
            if t.get("name") == name:
                self.selected_task = t; break
        self.accept()

def select_task_for_profile(profile_name: str):
    """
    Модальный выбор задания. Ручной event-loop для совместимости с arcade/macOS.
    """
    import time
    from PyQt6.QtGui import QGuiApplication

    app = QApplication.instance()
    created = False
    if app is None:
        app = QApplication(sys.argv)
        created = True
        app.setQuitOnLastWindowClosed(False)

    dlg = TaskPickerDialog(profile_name)
    dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
    dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
    dlg.show()
    app.processEvents()
    try:
        dlg.raise_(); dlg.activateWindow()
        if QGuiApplication.focusWindow() is None:
            app.processEvents(); dlg.activateWindow()
    except Exception:
        pass

    selected = None
    accepted = {"done": False}
    def _on_accept():
        nonlocal selected
        selected = dlg.selected_task
        accepted["done"] = True
    def _on_reject():
        accepted["done"] = True

    dlg.accepted.connect(_on_accept)
    dlg.rejected.connect(_on_reject)

    while dlg.isVisible() and not accepted["done"]:
        app.processEvents()
        time.sleep(0.01)

    try:
        if dlg.isVisible():
            dlg.close()
    except Exception:
        pass

    if created:
        app.processEvents(); app.quit()

    return selected
