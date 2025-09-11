import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

# GUI
from .guui.profile_select import ProfileSelectWindow
from .guui.main_window import MainWindow
from .guui.profiles import get_audio

# Игра
from .game.run_game import run_gromik_game_with_task


class LauncherMainWindow(MainWindow):
    """
    Наследуемся от MainWindow и переопределяем start_game:
    закрываем GUI и стартуем игру с нужными параметрами.
    """
    def start_game(self):
        task = self._get_selected_task()
        if not task:
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return

        audio = get_audio(self.profile_name)
        mic = audio.get("mic_device_index")
        if mic is None:
            QMessageBox.information(self, "Микрофон не выбран",
                                    "Откройте «Настройки аудио» и выберите микрофон.")
            return

        # Собираем payload для игры. Игра читает mic_device_index из task.
        payload = dict(task)
        payload["mic_device_index"] = mic
        # Можно передать и порог (игнорируется текущей логикой, но не мешает):
        payload["silence_threshold_db"] = audio.get("silence_threshold_db", 50.0)
        payload["profile_name"] = self.profile_name

        self.close()
        QTimer.singleShot(0, lambda: run_gromik_game_with_task(payload))


class LauncherProfileSelect(ProfileSelectWindow):
    """
    Подменяем открытие профиля так, чтобы создавался наш LauncherMainWindow.
    """
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
