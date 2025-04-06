import sys
from PyQt6.QtWidgets import QApplication
from guiconfig import MainWindow
from birdint import run_game_with_task  # это твоя игра

# расширяем MainWindow, чтобы переопределить кнопку "Старт игры"
class GameLauncher(MainWindow):
    def start_game(self):
        task = self.get_selected_task()
        if not task:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return
        self.close()  # закрываем окно GUI перед стартом игры
        run_game_with_task(task)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GameLauncher()
    window.show()
    sys.exit(app.exec())
