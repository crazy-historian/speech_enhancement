from __future__ import annotations

import sys
import os
import json
import subprocess
import tempfile
import platform
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

# GUI импорты
from .guui.profile_select import ProfileSelectWindow
from .guui.main_window import MainWindow
from .guui.profiles import get_audio


# ==========================================================
# 1. Если файл запущен с флагом --run-game → запускаем игру
# ==========================================================
if "--run-game" in sys.argv:
    from .game.run_game import run_bowling_game_with_task

    payload_path = sys.argv[-1]
    with open(payload_path, encoding="utf-8") as f:
        payload = json.load(f)

    run_bowling_game_with_task(payload)
    sys.exit(0)


# ==========================================================
# 2. Основные классы GUI
# ==========================================================
class LauncherMainWindow(MainWindow):
    """
    Главное окно запуска.
    На macOS — игра стартует в отдельном процессе,
    на Windows/Linux — прямо из Qt.
    """

    def start_game(self):
        # Проверяем выбранное задание
        task = self._get_selected_task()
        if not task:
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return

        # Проверяем аудио-настройки
        audio = get_audio(self.profile_name)
        mic = audio.get("mic_device_index")
        if mic is None:
            QMessageBox.information(
                self,
                "Микрофон не выбран",
                "Откройте «Настройки аудио» и выберите микрофон."
            )
            return

        # Формируем payload
        payload = dict(task)
        payload["mic_device_index"] = mic
        payload["silence_threshold_db"] = audio.get("silence_threshold_db", 50.0)
        payload["profile_name"] = self.profile_name

        # Сохраняем payload во временный JSON (текстовый режим!)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name

        # Закрываем GUI
        self.close()

        # macOS требует отдельный процесс
        if platform.system() == "Darwin":
            # ✅ Запускаем как модуль, чтобы относительные импорты работали
            subprocess.Popen([
                sys.executable,
                "-m",
                "games.bowling.launcher",
                "--run-game",
                tmp_path
            ])
        else:
            # ✅ На Windows/Linux можно запускать напрямую
            from .game.run_game import run_bowling_game_with_task
            QTimer.singleShot(0, lambda: run_bowling_game_with_task(payload))


class LauncherProfileSelect(ProfileSelectWindow):
    """Подменяем открытие профиля, чтобы создавался LauncherMainWindow."""
    def open_selected_profile(self):
        name = self._current_name()
        if not name:
            QMessageBox.information(self, "Открыть профиль", "Выберите профиль.")
            return
        self.hide()
        self.main = LauncherMainWindow(profile_name=name)
        self.main.show()
        self.close()


# ==========================================================
# 3. Точка входа в приложение
# ==========================================================
def main():
    app = QApplication(sys.argv)
    w = LauncherProfileSelect()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
