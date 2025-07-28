import sys
import json
from pathlib import Path
from PyQt6.QtCore import Qt 

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QPushButton, QListWidget, QLineEdit, QHBoxLayout, QFileDialog, QMessageBox,
    QStackedWidget, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox, QListWidgetItem, 
)


# === МОДЕЛИ ===
class Task:
    def __init__(self, name, frequency, duration, max_out_of_zone, description):
        self.name = name
        self.frequency = frequency  # может быть строка ("normal") или список [min, max]
        self.duration = duration
        self.max_out_of_zone = max_out_of_zone
        self.description = description

    def to_dict(self):
        return {
            "name": self.name,
            "frequency": self.frequency,
            "duration": self.duration,
            "max_out_of_zone": self.max_out_of_zone,
            "description": self.description
        }

    @staticmethod
    def from_dict(data):
        return Task(
            data["name"],
            data["frequency"],
            data["duration"],
            data["max_out_of_zone"],
            data["description"]
        )


class Profile:
    def __init__(self, name, settings=None, tasks=None):
        self.name = name
        self.settings = settings if settings else {
            "pitch_ranges": {
                "low": [150, 199],
                "normal": [200, 250],
                "high": [251, 300]
            },
            "volume_threshold_db": 50,
            "base_duration": 4.0,
            "max_out_of_zone": 0.5,
            "blocks_to_silent": 2
        }
        self.tasks = tasks if tasks else []

    def to_dict(self):
        return {
            "name": self.name,
            "settings": self.settings,
            "tasks": [task.to_dict() for task in self.tasks]
        }

    @staticmethod
    def from_dict(data):
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        return Profile(data["name"], data["settings"], tasks)

    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return Profile.from_dict(json.load(f))


# === UI ===
class StartScreen(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        layout = QVBoxLayout()

        self.profile_name_input = QLineEdit()
        self.profile_name_input.setPlaceholderText("Введите имя нового профиля")
        layout.addWidget(self.profile_name_input)

        self.create_btn = QPushButton("Создать профиль")
        self.create_btn.clicked.connect(self.create_profile)
        layout.addWidget(self.create_btn)

        self.load_btn = QPushButton("Загрузить профиль")
        self.load_btn.clicked.connect(self.load_profile)
        layout.addWidget(self.load_btn)

        self.setLayout(layout)

    def create_profile(self):
        name = self.profile_name_input.text()
        if not name:
            QMessageBox.warning(self, "Ошибка", "Введите имя профиля")
            return
        profile = Profile(name)
        editor = ProfileEditor(self.stacked_widget, profile)
        self.stacked_widget.addWidget(editor)
        self.stacked_widget.setCurrentWidget(editor)

    def load_profile(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть профиль", "", "JSON Files (*.json)")
        if path:
            profile = Profile.load(path)
            editor = ProfileEditor(self.stacked_widget, profile, path)
            self.stacked_widget.addWidget(editor)
            self.stacked_widget.setCurrentWidget(editor)


class TaskEditor(QWidget):
    def __init__(self, parent_editor, profile, task=None, task_index=None):
        super().__init__()
        self.parent_editor = parent_editor
        self.profile = profile
        self.task = task
        self.task_index = task_index

        layout = QVBoxLayout()

        self.name_input = QLineEdit()
        layout.addWidget(QLabel("Название задания:"))
        layout.addWidget(self.name_input)

        self.freq_input = QComboBox()
        self.freq_input.addItem(f"low ({self.profile.settings['pitch_ranges']['low'][0]}–{self.profile.settings['pitch_ranges']['low'][1]})")
        self.freq_input.addItem(f"normal ({self.profile.settings['pitch_ranges']['normal'][0]}–{self.profile.settings['pitch_ranges']['normal'][1]})")
        self.freq_input.addItem(f"high ({self.profile.settings['pitch_ranges']['high'][0]}–{self.profile.settings['pitch_ranges']['high'][1]})")
        self.freq_input.addItem("custom")
        layout.addWidget(QLabel("Частотная зона:"))
        layout.addWidget(self.freq_input)

        self.custom_freq_input = QLineEdit()
        self.custom_freq_input.setPlaceholderText("Например: 180-220")
        self.custom_freq_input.setVisible(False)
        layout.addWidget(self.custom_freq_input)

        self.freq_input.currentTextChanged.connect(self.toggle_custom_freq)

        self.duration_input = QDoubleSpinBox()
        self.duration_input.setValue(self.profile.settings["base_duration"])
        layout.addWidget(QLabel("Длительность (сек):"))
        layout.addWidget(self.duration_input)

        self.max_out_input = QDoubleSpinBox()
        self.max_out_input.setValue(self.profile.settings["max_out_of_zone"])
        layout.addWidget(QLabel("Макс. пребывание за пределами (сек):"))
        layout.addWidget(self.max_out_input)

        self.desc_input = QTextEdit()
        layout.addWidget(QLabel("Описание задания:"))
        layout.addWidget(self.desc_input)

        self.save_btn = QPushButton("Сохранить задание")
        self.save_btn.clicked.connect(self.save_task)
        layout.addWidget(self.save_btn)

        self.back_btn = QPushButton("Назад")
        self.back_btn.clicked.connect(self.back)
        layout.addWidget(self.back_btn)

        self.setLayout(layout)

        if self.task:
            self.load_task_data()

    def toggle_custom_freq(self, text):
        self.custom_freq_input.setVisible("custom" in text)

    def load_task_data(self):
        self.name_input.setText(self.task.name)
        if isinstance(self.task.frequency, list):
            self.freq_input.setCurrentText("custom")
            self.custom_freq_input.setText(f"{self.task.frequency[0]}-{self.task.frequency[1]}")
            self.custom_freq_input.setVisible(True)
        else:
            index = self.freq_input.findText(self.task.frequency)

            if index >= 0:
                self.freq_input.setCurrentIndex(index)
        self.duration_input.setValue(self.task.duration)
        self.max_out_input.setValue(self.task.max_out_of_zone)
        self.desc_input.setPlainText(self.task.description)

    def save_task(self):
        freq_text = self.freq_input.currentText()
        if "custom" in freq_text:
            try:
                parts = self.custom_freq_input.text().split("-")
                frequency = [int(parts[0]), int(parts[1])]
            except:
                QMessageBox.warning(self, "Ошибка", "Неверный формат частоты. Используй: 180-220")
                return
        else:
            frequency = freq_text.split()[0]  # берём "low", "normal" и т.п.

        updated_task = Task(
            name=self.name_input.text(),
            frequency=frequency,
            duration=self.duration_input.value(),
            max_out_of_zone=self.max_out_input.value(),
            description=self.desc_input.toPlainText()
        )
        if self.task_index is not None:
            self.profile.tasks[self.task_index] = updated_task
        else:
            self.profile.tasks.append(updated_task)
        self.back()

    def back(self):
        self.parent_editor.update_task_list()
        self.parent_editor.stacked_widget.setCurrentWidget(self.parent_editor)


class ProfileEditor(QWidget):
    def __init__(self, stacked_widget, profile, filepath=None):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.profile = profile
        self.filepath = filepath

        layout = QVBoxLayout()

        layout.addWidget(QLabel(f"Имя профиля: {profile.name}"))

        self.low_range = QLineEdit(f"{profile.settings['pitch_ranges']['low'][0]}-{profile.settings['pitch_ranges']['low'][1]}")
        layout.addWidget(QLabel("Частота низкая (low):"))
        layout.addWidget(self.low_range)

        self.norm_range = QLineEdit(f"{profile.settings['pitch_ranges']['normal'][0]}-{profile.settings['pitch_ranges']['normal'][1]}")
        layout.addWidget(QLabel("Частота нормальная (normal):"))
        layout.addWidget(self.norm_range)

        self.high_range = QLineEdit(f"{profile.settings['pitch_ranges']['high'][0]}-{profile.settings['pitch_ranges']['high'][1]}")
        layout.addWidget(QLabel("Частота высокая (high):"))
        layout.addWidget(self.high_range)

        self.volume_threshold = QDoubleSpinBox()
        self.volume_threshold.setValue(profile.settings["volume_threshold_db"])
        layout.addWidget(QLabel("Порог громкости (дБ):"))
        layout.addWidget(self.volume_threshold)

        self.duration = QDoubleSpinBox()
        self.duration.setValue(profile.settings["base_duration"])
        layout.addWidget(QLabel("Базовая длительность (сек):"))
        layout.addWidget(self.duration)

        self.max_out = QDoubleSpinBox()
        self.max_out.setValue(profile.settings["max_out_of_zone"])
        layout.addWidget(QLabel("Максимальное пребывание за пределом (сек):"))
        layout.addWidget(self.max_out)

        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self.edit_task)

        layout.addWidget(QLabel("Задания профиля:"))
        layout.addWidget(self.task_list)

        self.new_task_btn = QPushButton("Добавить задание")
        self.new_task_btn.clicked.connect(self.create_task)
        layout.addWidget(self.new_task_btn)

        self.save_btn = QPushButton("Сохранить профиль")
        self.save_btn.clicked.connect(self.save_profile)
        layout.addWidget(self.save_btn)

        self.start_btn = QPushButton("Старт игры")
        self.start_btn.clicked.connect(self.start_game)
        layout.addWidget(self.start_btn)

        self.setLayout(layout)
        self.update_task_list()

    def update_task_list(self):
        self.task_list.clear()
        for i, task in enumerate(self.profile.tasks):
            item = QListWidgetItem(f"{task.name} — {task.description}")
            item.setData(1, i)
            self.task_list.addItem(item)

    def create_task(self):
        editor = TaskEditor(self, self.profile)
        self.stacked_widget.addWidget(editor)
        self.stacked_widget.setCurrentWidget(editor)

    def edit_task(self, item):
        index = item.data(1)
        task = self.profile.tasks[index]
        editor = TaskEditor(self, self.profile, task, index)
        self.stacked_widget.addWidget(editor)
        self.stacked_widget.setCurrentWidget(editor)

    def save_profile(self):
        def parse_range(rng):
            parts = rng.split("-")
            return [int(parts[0]), int(parts[1])]

        self.profile.settings["pitch_ranges"]["low"] = parse_range(self.low_range.text())
        self.profile.settings["pitch_ranges"]["normal"] = parse_range(self.norm_range.text())
        self.profile.settings["pitch_ranges"]["high"] = parse_range(self.high_range.text())
        self.profile.settings["volume_threshold_db"] = self.volume_threshold.value()
        self.profile.settings["base_duration"] = self.duration.value()
        self.profile.settings["max_out_of_zone"] = self.max_out.value()

        path, _ = QFileDialog.getSaveFileName(self, "Сохранить профиль", f"{self.profile.name}.json", "JSON Files (*.json)")
        if path:
            self.profile.save(path)
            QMessageBox.information(self, "Сохранено", "Профиль сохранён.")
    
    def start_game(self):
        current_item = self.task_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Ошибка", "Выберите задание для запуска")
            return
        index = current_item.data(1)
        task = self.profile.tasks[index]

        # Сохраняем выбор в главном окне
        self.stacked_widget.parent().selected_profile = self.profile
        self.stacked_widget.parent().selected_task = task
        self.stacked_widget.parent().close()  # Закрыть PyQt окно


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_profile = None
        self.selected_task = None
        self.setWindowTitle("Голосовой тренажёр — Профили и задания")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.start_screen = StartScreen(self.stack)
        self.stack.addWidget(self.start_screen)
        self.stack.setCurrentWidget(self.start_screen)

def select_task_from_profile(profile):
    from PyQt6.QtWidgets import QDialog

    class TaskSelector(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Выбор задания")
            self.setGeometry(100, 100, 400, 300)
            self.selected_task = None

            layout = QVBoxLayout()
            self.list_widget = QListWidget()

            for task in profile.tasks:
                item = QListWidgetItem(f"{task.name} — {task.description}")
                item.setData(1, task)
                self.list_widget.addItem(item)

            layout.addWidget(self.list_widget)

            select_button = QPushButton("Выбрать")
            select_button.clicked.connect(self.confirm_selection)
            layout.addWidget(select_button)

            self.setLayout(layout)

        def confirm_selection(self):
            item = self.list_widget.currentItem()
            if item:
                self.selected_task = item.data(1)
                self.accept()  # это ключевой момент

    app = QApplication.instance()
    owns_app = False

    if not app:
        app = QApplication(sys.argv)
        owns_app = True

    dialog = TaskSelector()
    result = dialog.exec()

    if owns_app:
        app.quit()

    if result == QDialog.DialogCode.Accepted and dialog.selected_task:
        return dialog.selected_task
    return None
def launch_profile_and_task():
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

    return window.selected_profile, window.selected_task



# === ЗАПУСК ===
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

