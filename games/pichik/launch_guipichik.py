# pitch_launcher.py
import sys
from PyQt6.QtWidgets import QApplication
from guipichik import MainWindow, load_pitch_config
from game.run_game import run_pitch_game_with_task

class PitchGameLauncher(MainWindow):
    def start_game(self):
        task = self.get_selected_task()
        if not task:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return
        
        # Сохраняем выбранный микрофон (device_index) прямо в task, чтобы
        # "run_pitch_game_with_task" знал, какой device использовать
        config = load_pitch_config()
        dev_index = config.get("mic_device_index", None)
        task["mic_device_index"] = dev_index

        self.close()  # Закрываем окно GUI
        run_pitch_game_with_task(task)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PitchGameLauncher()
    window.show()
    sys.exit(app.exec())
