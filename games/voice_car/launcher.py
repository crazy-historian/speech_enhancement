# -*- coding: utf-8 -*-
import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

# GUI
from .guui.profile_select import ProfileSelectWindow
from .guui.main_window import MainWindow
from .guui.profiles import get_audio, get_settings

# Игра
from .game.run_game import run_game_with_profile_task


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
            QMessageBox.information(
                self, "Микрофон не выбран",
                "Откройте «Настройки аудио» и выберите микрофон."
            )
            return

        # Сбор профиля в dict (как ожидает run_game_with_profile_task)
        profile = {
            "name": self.profile_name,
            "audio": audio,                       # содержит mic_device_index и silence_threshold_db
            "settings": get_settings(self.profile_name)  # pitch_ranges, blocks_to_silent, etc.
        }

        # Закрываем GUI и в следующем тике запускаем игру
        self.close()
        QTimer.singleShot(0, lambda: run_game_with_profile_task(profile, task))


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
