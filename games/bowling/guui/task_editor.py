from typing import Optional
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIntValidator
from .profiles import upsert_task

class TaskEditor(QWidget):
    """ Простой редактор задания: Name, Duration, Text. """
    def __init__(self, task=None, on_save=None, on_close=None, profile_name: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Редактор задания (Bowling)")
        self.setMinimumWidth(480)
        self.profile_name = profile_name
        self.task = task or {}
        self.on_save = on_save
        self.on_close = on_close
        self._saved = False

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Название задания:"))
        self.name_edit = QLineEdit(str(self.task.get("name", "")))
        layout.addWidget(self.name_edit)

        layout.addWidget(QLabel("Время непрерывного говорения (сек):"))
        self.duration_line = QLineEdit(str(self.task.get("duration", "2")))
        self.duration_line.setValidator(QIntValidator(1, 3600, self))
        layout.addWidget(self.duration_line)

        layout.addWidget(QLabel("Текст задания:"))
        self.text_line = QLineEdit(str(self.task.get("text", "МА")))
        layout.addWidget(self.text_line)

        btns = QHBoxLayout()
        btn_save = QPushButton("Сохранить")
        btn_back = QPushButton("Назад")
        btns.addWidget(btn_save); btns.addWidget(btn_back)
        layout.addLayout(btns)

        btn_save.clicked.connect(self.save_task)
        btn_back.clicked.connect(self.close)

    def save_task(self):
        def int_or(le: QLineEdit, default: int) -> int:
            txt = le.text().strip()
            return int(txt) if txt else default

        name = self.name_edit.text().strip()
        new_task = {
            "name": name or "Без названия",
            "duration": int_or(self.duration_line, 2),
            "text": self.text_line.text().strip() or "МА",
        }
        if not self.profile_name:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Профиль не задан", "Редактор открыт без профиля")
            return
        upsert_task(self.profile_name, new_task)
        self._saved = True
        if callable(self.on_save):
            QTimer.singleShot(0, self.on_save)
        self.close()

    def closeEvent(self, event):
        if not self._saved and callable(self.on_close):
            QTimer.singleShot(0, self.on_close)
        super().closeEvent(event)
