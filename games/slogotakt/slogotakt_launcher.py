from PyQt6.QtWidgets import QApplication
from pathlib import Path
import json
import sys

from guislogotakt import SyllableConfigWindow  # имя файла, где находится GUI
from game.core_game import VoiceArcadeGame

def load_config():
    config_path = Path("games/slogotakt/components/syllable_config.json")
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {"audio": {}, "tasks": []}

def run_gui_and_get_task():
    app = QApplication(sys.argv)
    window = SyllableConfigWindow()
    window.show()
    app.exec()

    # 💣 Полностью убиваем QApplication, чтобы освободить event loop
    app.quit()
    del app

    if window.should_start_game and window.selected_task:
        return window.selected_task
    return None

if __name__ == "__main__":
    config = load_config()
    task = run_gui_and_get_task()

    if not task:
        print("Игра не запущена.")
        sys.exit(0)

    silence_threshold_db = config.get("audio", {}).get("silence_threshold_db", 45.0)
    print('Запуск игры')

    # Запуск игры
    game = VoiceArcadeGame(task, silence_threshold_db)
    game.setup()
    game.run()
