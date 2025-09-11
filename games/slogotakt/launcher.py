import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

# GUI
from .guui.profile_select import ProfileSelectWindow
from .guui.main_window import MainWindow
from .guui.profiles import get_audio

# Игра
from .game.run_game import run_slogotakt_with_task


class LauncherMainWindow(MainWindow):
    """
    Наследуемся от MainWindow и переопределяем start_game:
    закрываем GUI и стартуем игру с нужными параметрами.
    """
    def start_game(self):
        it = self.task_list.currentItem()
        if not it:
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return
        # найдём task по имени
        name = it.text()
        task = None
        for t in self._find_task_by_name(name),:
            task = t
            break
        if task is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось найти выбранное задание")
            return

        audio = get_audio(self.profile_name)
        mic = audio.get("mic_device_index")
        if mic is None:
            QMessageBox.information(self, "Микрофон не выбран",
                                    "Откройте «Настройки аудио» и выберите микрофон.")
            return

        # Собираем payload для игры. Игра читает mic_device_index и др. из task.
        payload = dict(task)
        payload["mic_device_index"] = mic
        payload["silence_threshold_db"] = audio.get("silence_threshold_db", 50.0)
        payload["profile_name"] = self.profile_name

        self.close()
        QTimer.singleShot(0, lambda: run_slogotakt_with_task(payload))


class LauncherProfileSelect(ProfileSelectWindow):
    """Подменяем открытие профиля так, чтобы создавался LauncherMainWindow."""
    def open_selected_profile(self):
        name = self._current_name()
        if not name:
            QMessageBox.information(self, "Открыть профиль", "Выберите профиль.")
            return
        self.hide()
        self.main = LauncherMainWindow(profile_name=name)
        self.main.show()
        self.close()


def main():
    app = QApplication(sys.argv)
    w = LauncherProfileSelect()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()