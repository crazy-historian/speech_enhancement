import sys
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox,
    QListWidget, QApplication, QStackedWidget
)
from PyQt6.QtCore import QTimer

from .audio_settings import AudioSettingsWindow
from .task_editor import TaskEditor
from .profiles import get_tasks, get_audio, set_tasks

class MainWindow(QWidget):
    """
    Главное окно VoiceCar: список задач, редактор задания, аудионастройки.
    """
    def __init__(self, profile_name: str):
        super().__init__()
        self.profile_name = profile_name
        self.setWindowTitle(f"VoiceCar — {self.profile_name}")
        self.setMinimumWidth(460)

        self._is_closing = False
        self._closed = False
        self._profile_window = None
        self.editor_page: TaskEditor | None = None

        self.stacked = QStackedWidget(self)

        # --- страница 0: список задач ---
        self.main_page = QWidget(self)
        main_layout = QVBoxLayout(self.main_page)

        main_layout.addWidget(QLabel(f"Профиль: {self.profile_name}"))
        main_layout.addWidget(QLabel("Задания:"))

        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self._edit_task_from_list)
        main_layout.addWidget(self.task_list)

        btns = QHBoxLayout()
        self.btn_back_profiles = QPushButton("← К профилям")
        self.btn_audio = QPushButton("Настройки аудио")
        self.btn_add = QPushButton("Добавить")
        self.btn_delete = QPushButton("Удалить")
        self.btn_start = QPushButton("Старт игры")

        self.btn_back_profiles.clicked.connect(self.back_to_profiles)
        self.btn_audio.clicked.connect(self.show_audio_settings)
        self.btn_add.clicked.connect(self.open_new_task_editor)
        self.btn_delete.clicked.connect(self.delete_selected_task)
        self.btn_start.clicked.connect(self.start_game)

        for b in (self.btn_back_profiles, self.btn_audio, self.btn_add, self.btn_delete, self.btn_start):
            btns.addWidget(b)
        main_layout.addLayout(btns)

        # --- страница 1: аудио ---
        self.audio_page = AudioSettingsWindow(profile_name=self.profile_name, go_back=self.show_main)

        self.stacked.addWidget(self.main_page)   # idx 0
        self.stacked.addWidget(self.audio_page)  # idx 1

        root = QVBoxLayout(self)
        root.addWidget(self.stacked)
        self.setLayout(root)

        self.show_main()

    # --------- utils ---------
    def _cleanup_editor(self):
        ed = self.editor_page
        if not ed:
            return
        try:
            ed.on_save = None
            ed.on_close = None
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

    # --------- навигация ---------
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

    # --------- задачи ---------
    def load_tasks(self):
        self.task_list.clear()
        for t in get_tasks(self.profile_name):
            self.task_list.addItem(t.get("name", "(без названия)"))

    def _get_selected_task(self):
        it = self.task_list.currentItem()
        if not it:
            return None
        name = it.text()
        for t in get_tasks(self.profile_name):
            if t.get("name") == name:
                return t
        return None

    def _edit_task_from_list(self, item):
        name = item.text()
        task = None
        for t in get_tasks(self.profile_name):
            if t.get("name") == name:
                task = t; break
        self.open_task_editor(task)

    def open_new_task_editor(self):
        self.open_task_editor(None)

    def open_task_editor(self, task):
        self._cleanup_editor()
        self.editor_page = TaskEditor(
            task=task,
            profile_name=self.profile_name,
            on_save=self._on_editor_saved,
            on_close=self._on_editor_closed
        )
        try:
            self.stacked.addWidget(self.editor_page)
            self.stacked.setCurrentWidget(self.editor_page)
        except Exception:
            pass

    def _on_editor_saved(self):
        if self._is_closing or self._closed:
            return
        self._cleanup_editor()
        self.show_main()

    def _on_editor_closed(self):
        if self._is_closing or self._closed:
            return
        self._cleanup_editor()
        self.show_main()

    def delete_selected_task(self):
        it = self.task_list.currentItem()
        if not it:
            QMessageBox.information(self, "Удаление", "Выберите задание."); return
        name = it.text()
        from PyQt6.QtWidgets import QMessageBox as MB
        if MB.question(self, "Подтверждение", f"Удалить «{name}»?",
                       MB.StandardButton.Yes | MB.StandardButton.No,
                       MB.StandardButton.No) != MB.StandardButton.Yes:
            return
        tasks = [t for t in get_tasks(self.profile_name) if t.get("name") != name]
        set_tasks(self.profile_name, tasks)
        self.load_tasks()

    # --------- системные ---------
    def closeEvent(self, e):
        self._closed = True
        try: self._cleanup_editor()
        except Exception: pass
        super().closeEvent(e)

    # --------- запуск игры ---------
    def start_game(self):
        task = self._get_selected_task()
        if not task:
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска."); return

        audio = get_audio(self.profile_name)
        if audio.get("mic_device_index") is None:
            QMessageBox.information(self, "Микрофон", "Выберите микрофон в «Настройки аудио»."); return

        print(f"[VoiceCar RUN] profile={self.profile_name} task={task} mic={audio.get('mic_device_index')} thr={audio.get('silence_threshold_db')}")
        # здесь можешь интегрировать реальный запуск игры
        # from games.voice_car import VoiceArcadeGame
        # VoiceArcadeGame(task, profile_name=self.profile_name).run()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    prof = sys.argv[1] if len(sys.argv) > 1 else "Demo"
    w = MainWindow(profile_name=prof)
    w.show()
    sys.exit(app.exec())
