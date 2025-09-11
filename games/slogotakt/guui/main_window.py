from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QListWidget, QStackedWidget
)
from PyQt6.QtCore import QTimer

from .audio_settings import AudioSettingsWindow
from .task_editor import TaskEditor
from .profiles import get_tasks, get_audio, set_tasks


class MainWindow(QWidget):
    """Главное окно slogotakt (QStackedWidget):
    1) Главная со списком заданий
    2) Настройки аудио
    3) Редактор задания (создаётся по требованию)
    """
    def __init__(self, profile_name: str):
        super().__init__()
        self.profile_name = profile_name
        self.setWindowTitle(f"Настройка игры на слоги — {self.profile_name}")
        self.setMinimumWidth(440)
        self._is_closing = False
        self._closed = False
        self._profile_window = None
        self.editor_page: Optional[TaskEditor] = None

        self.stacked = QStackedWidget(self)

        # Главная страница
        self.main_page = QWidget(self)
        main_layout = QVBoxLayout(self.main_page)
        main_layout.addWidget(QLabel(f"Профиль: {self.profile_name}"))
        main_layout.addWidget(QLabel("Выберите задание:"))
        self.task_list = QListWidget(); self.task_list.itemDoubleClicked.connect(self._edit_task_from_list)
        main_layout.addWidget(self.task_list)

        btns = QHBoxLayout()
        self.btn_back_profiles = QPushButton("← К профилям")
        self.btn_audio = QPushButton("Настройки аудио")
        self.btn_add = QPushButton("Добавить задание")
        self.btn_delete = QPushButton("Удалить задание")
        self.btn_start = QPushButton("Старт игры")

        self.btn_back_profiles.clicked.connect(self.back_to_profiles)
        self.btn_audio.clicked.connect(self.show_audio_settings)
        self.btn_add.clicked.connect(self.open_new_task_editor)
        self.btn_delete.clicked.connect(self.delete_selected_task)
        self.btn_start.clicked.connect(self.start_game)

        for b in [self.btn_back_profiles, self.btn_audio, self.btn_add, self.btn_delete, self.btn_start]:
            btns.addWidget(b)
        main_layout.addLayout(btns)

        # Настройки аудио
        self.audio_page = AudioSettingsWindow(profile_name=self.profile_name, go_back=self.show_main)

        self.stacked.addWidget(self.main_page)  # 0
        self.stacked.addWidget(self.audio_page) # 1

        root = QVBoxLayout(self); root.addWidget(self.stacked); self.setLayout(root)
        self.show_main()

    # --- сервис ---
    def _cleanup_editor(self):
        ed = self.editor_page
        if ed is None:
            return
        try:
            ed.on_save = None; ed.on_close = None
        except Exception:
            pass
        try:
            idx = self.stacked.indexOf(ed)
            if idx != -1:
                self.stacked.removeWidget(ed)
        except Exception:
            pass
        try:
            ed.deleteLater()
        except Exception:
            pass
        self.editor_page = None

    # --- навигация ---
    def show_main(self):
        if self._closed:
            return
        self.load_tasks()
        try:
            self.stacked.setCurrentWidget(self.main_page)
        except Exception:
            pass
        try:
            self.audio_page._load()
        except Exception:
            pass

    def back_to_profiles(self):
        self._is_closing = True
        self._cleanup_editor()
        def _open_profiles():
            try:
                from .profile_select import ProfileSelectWindow
                self._profile_window = ProfileSelectWindow()
                self._profile_window.show()
            except Exception:
                pass
        QTimer.singleShot(0, _open_profiles)
        self.close()

    def show_audio_settings(self):
        try:
            self.stacked.setCurrentWidget(self.audio_page)
        except Exception:
            pass

    # --- задачи ---
    def load_tasks(self):
        self.task_list.clear()
        for t in get_tasks(self.profile_name):
            self.task_list.addItem(t.get("name", "(без названия)"))

    def _find_task_by_name(self, name: str):
        for t in get_tasks(self.profile_name):
            if t.get("name") == name:
                return t
        return None

    def _edit_task_from_list(self, item):
        self.open_task_editor(self._find_task_by_name(item.text()))

    def open_new_task_editor(self):
        self.open_task_editor(None)

    def open_task_editor(self, task):
        self._cleanup_editor()
        self.editor_page = TaskEditor(task=task, profile_name=self.profile_name, on_save=self._on_editor_saved, on_close=self._on_editor_closed)
        try:
            self.stacked.addWidget(self.editor_page)
            self.stacked.setCurrentWidget(self.editor_page)
        except Exception:
            pass

    def _on_editor_saved(self):
        if self._is_closing or self._closed:
            return
        self._cleanup_editor(); self.show_main()

    def _on_editor_closed(self):
        if self._is_closing or self._closed:
            return
        self._cleanup_editor(); self.show_main()

    def delete_selected_task(self):
        from PyQt6.QtWidgets import QMessageBox
        it = self.task_list.currentItem()
        if not it:
            QMessageBox.information(self, "Удаление задания", "Выберите задание для удаления.")
            return
        name = it.text()
        reply = QMessageBox.question(
            self, "Подтверждение", f"Точно удалить задание «{name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        tasks = get_tasks(self.profile_name)
        new_tasks = [t for t in tasks if t.get("name") != name]
        if len(new_tasks) == len(tasks):
            QMessageBox.information(self, "Удаление задания", "Задание не найдено."); return
        set_tasks(self.profile_name, new_tasks)
        self.load_tasks()

    def closeEvent(self, event):
        self._closed = True
        try:
            self._cleanup_editor()
        except Exception:
            pass
        super().closeEvent(event)

    # --- старт игры ---
    def start_game(self):
        from PyQt6.QtWidgets import QMessageBox
        it = self.task_list.currentItem()
        if not it:
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return
        task = self._find_task_by_name(it.text())
        audio = get_audio(self.profile_name)
        if audio.get("mic_device_index") is None:
            QMessageBox.information(self, "Микрофон не выбран", "Откройте «Настройки аудио» и выберите микрофон.")
            return
        print(f"[RUN slogotakt] profile={self.profile_name}; task={task}; mic={audio.get('mic_device_index')} thr={audio.get('silence_threshold_db')}")
        self.close()
        # TODO: подключи свой раннер игры здесь
        # from ..game.run_game import run_slogotakt_with_task
        # run_slogotakt_with_task(profile_name=self.profile_name, task=task)