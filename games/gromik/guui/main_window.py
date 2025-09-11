from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox,
    QListWidget, QApplication, QStackedWidget
)
from PyQt6.QtCore import QTimer
import sys

from .audio_settings import AudioSettingsWindow
from .task_editor import TaskEditor
from .profiles import get_tasks, get_audio, set_tasks  # <-- добавил set_tasks


class MainWindow(QWidget):
    """
    Pitch — главное окно: один QStackedWidget с 3 страницами:
      1) Главная (список задач + кнопки)
      2) Настройки аудио (AudioSettingsWindow)
      3) Редактор задания (TaskEditor) — создаём по требованию
    """
    def __init__(self, profile_name: str):
        super().__init__()
        self.profile_name = profile_name
        self.setWindowTitle(f"Настройки игры (Pitch) — {self.profile_name}")
        self.setMinimumWidth(420)

        self._is_closing = False
        self._closed = False
        self._profile_window = None
        self.editor_page: TaskEditor | None = None

        # ---- стек страниц ----
        self.stacked = QStackedWidget(self)

        # 1) Главная страница
        self.main_page = QWidget(self)
        main_layout = QVBoxLayout(self.main_page)

        main_layout.addWidget(QLabel(f"Профиль: {self.profile_name}"))
        main_layout.addWidget(QLabel("Выберите задание:"))

        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self._edit_task_from_list)
        main_layout.addWidget(self.task_list)

        btns = QHBoxLayout()
        self.btn_back_profiles = QPushButton("← К профилям")
        self.btn_audio = QPushButton("Настройки аудио")
        self.btn_add = QPushButton("Добавить задание")
        self.btn_delete = QPushButton("Удалить задание")   # <-- новая кнопка
        self.btn_start = QPushButton("Старт игры")

        self.btn_back_profiles.clicked.connect(self.back_to_profiles)
        self.btn_audio.clicked.connect(self.show_audio_settings)
        self.btn_add.clicked.connect(self.open_new_task_editor)
        self.btn_delete.clicked.connect(self.delete_selected_task)  # <-- обработчик
        self.btn_start.clicked.connect(self.start_game)

        btns.addWidget(self.btn_back_profiles)
        btns.addWidget(self.btn_audio)
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_delete)  # <-- добавили в панель
        btns.addWidget(self.btn_start)
        main_layout.addLayout(btns)

        # 2) Настройки аудио
        self.audio_page = AudioSettingsWindow(profile_name=self.profile_name, go_back=self.show_main)

        # Вставляем страницы в стек
        self.stacked.addWidget(self.main_page)   # idx 0
        self.stacked.addWidget(self.audio_page)  # idx 1

        # Корневой лэйаут
        root = QVBoxLayout(self)
        root.addWidget(self.stacked)
        self.setLayout(root)

        self.show_main()

    # ---------- вспомогательная уборка редактора ----------
    def _cleanup_editor(self):
        """Безопасно удалить вкладку редактора и заглушить её колбэки."""
        ed = self.editor_page
        if ed is None:
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

    # ---------- навигация ----------
    def show_main(self):
        if self._closed:
            return
        self.load_tasks()
        try:
            self.stacked.setCurrentWidget(self.main_page)
        except Exception:
            pass
        # обновим аудио-страницу
        try:
            self.audio_page._load()
        except Exception:
            pass

    def back_to_profiles(self):
        # ставим флаг и заранее чистим редактор, чтобы не прилетели его колбэки
        self._is_closing = True
        self._cleanup_editor()

        # открыть окно профилей после закрытия этого окна
        def _open_profiles():
            try:
                from .profile_select import ProfileSelectWindow  # локально, чтобы избежать циклов
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

    # ---------- задачи ----------
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
                task = t
                break
        self.open_task_editor(task)

    def open_new_task_editor(self):
        self.open_task_editor(None)

    def open_task_editor(self, task):
        self._cleanup_editor()  # уберём предыдущий, если был

        # создаём свежий редактор
        self.editor_page = TaskEditor(
            task=task,
            profile_name=self.profile_name,
            on_save=self._on_editor_saved,
            on_close=self._on_editor_closed
        )
        try:
            self.stacked.addWidget(self.editor_page)  # станет idx 2
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

    # ---------- удаление задания ----------
    def delete_selected_task(self):
        it = self.task_list.currentItem()
        if not it:
            QMessageBox.information(self, "Удаление задания", "Выберите задание для удаления.")
            return
        name = it.text()

        reply = QMessageBox.question(
            self, "Подтверждение",
            f"Точно удалить задание «{name}»?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        tasks = get_tasks(self.profile_name)
        new_tasks = [t for t in tasks if t.get("name") != name]
        if len(new_tasks) == len(tasks):
            QMessageBox.information(self, "Удаление задания", "Задание не найдено.")
            return

        set_tasks(self.profile_name, new_tasks)
        self.load_tasks()

    # ---------- системные ----------
    def closeEvent(self, event):
        self._closed = True
        # на всякий — подчистим редактор
        try:
            self._cleanup_editor()
        except Exception:
            pass
        super().closeEvent(event)

    # ---------- старт игры ----------
    def start_game(self):
        task = self._get_selected_task()
        if not task:
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return

        audio = get_audio(self.profile_name)
        if audio.get("mic_device_index") is None:
            QMessageBox.information(self, "Микрофон не выбран",
                                    "Откройте «Настройки аудио» и выберите микрофон.")
            return

        print(f"[RUN] Профиль: {self.profile_name}; Задание: {task}; "
              f"mic={audio.get('mic_device_index')}, thr={audio.get('silence_threshold_db')}")
        self.close()
        # TODO: run_pitch_game_with_task(task, profile_name=self.profile_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    profile = sys.argv[1] if len(sys.argv) > 1 else "Demo"
    w = MainWindow(profile_name=profile)
    w.show()
    sys.exit(app.exec())
