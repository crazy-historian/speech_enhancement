from __future__ import annotations

import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

# Относительные импорты GUI
from .guui.profile_select import ProfileSelectWindow
from .guui.main_window import MainWindow
from .guui.profiles import get_audio

# Относительный импорт раннера игры
from .game.run_game import run_bowling_game_with_task


class LauncherMainWindow(MainWindow):
    """
    Наследуемся от MainWindow и переопределяем start_game:
    закрываем GUI и стартуем игру с нужными параметрами.
    """
    def start_game(self):
        task = self._get_selected_task()
        if not task:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return

        audio = get_audio(self.profile_name)
        mic = audio.get("mic_device_index")
        if mic is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Микрофон не выбран",
                "Откройте «Настройки аудио» и выберите микрофон."
            )
            return

        # Формируем payload для игры
        payload = dict(task)
        payload["mic_device_index"] = mic
        payload["silence_threshold_db"] = audio.get("silence_threshold_db", 50.0)
        payload["profile_name"] = self.profile_name

        # Закрываем GUI и запускаем Arcade в следующем тике
        self.close()
        QTimer.singleShot(0, lambda: run_bowling_game_with_task(payload))


class LauncherProfileSelect(ProfileSelectWindow):
    """
    Подменяем открытие профиля так, чтобы создавался наш LauncherMainWindow.
    """
    def open_selected_profile(self):
        name = self._current_name()
        if not name:
            from PyQt6.QtWidgets import QMessageBox
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
