import sys
import json
import time
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit,
    QHBoxLayout, QListWidget, QStackedWidget, QComboBox,
    QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from audiochains.devices import AudioDevices
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

from boul import VoiceBowlingGame
import arcade

CONFIG_FILE = Path("profiles/bowling_config.json")
TASKS_FILE = Path("profiles/bowling_tasks.json")
BLOCKSIZE = 1024
RECORD_DURATION = 5


def load_config():
    if CONFIG_FILE.exists() and CONFIG_FILE.read_text(encoding="utf-8").strip():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {}


def save_config(data):
    CONFIG_FILE.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def load_tasks():
    if TASKS_FILE.exists() and TASKS_FILE.read_text(encoding="utf-8").strip():
        return json.loads(TASKS_FILE.read_text(encoding="utf-8"))
    return []


def save_tasks(tasks):
    TASKS_FILE.write_text(json.dumps(tasks, indent=4, ensure_ascii=False), encoding="utf-8")


class IntensityAnalyzer(QThread):
    result_ready = pyqtSignal(float, list, list)
    live_update = pyqtSignal(list, list)

    def __init__(self):
        super().__init__()
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
            stream.set_methods(UnpackRawInFloat32())
            start_time = time.time()
            times, intensities = [], []
            current_time = 0

            while time.time() - start_time < RECORD_DURATION and self._running:
                raw_data = stream.read(BLOCKSIZE)
                if not raw_data:
                    continue

                signal = stream.chain_of_methods(raw_data)
                sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)
                intensity_obj = sound.to_intensity()
                intensity_values = intensity_obj.values.T.flatten()
                avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else 0

                times.append(current_time)
                intensities.append(avg_intensity)
                current_time += BLOCKSIZE / 16000

                self.live_update.emit(times.copy(), intensities.copy())

            if intensities:
                overall_avg = np.mean(intensities)
                self.result_ready.emit(overall_avg, times, intensities)


class MatplotlibCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 2))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.ax.set_title("Интенсивность голоса")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Интенсивность (dB)")
        self.ax.grid(True)

    def plot_live(self, times, intensities):
        self.ax.clear()
        self.ax.plot(times, intensities, label="Интенсивность")
        self.ax.set_title("Интенсивность голоса")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Интенсивность (dB)")
        self.ax.grid(True)
        self.ax.legend()
        self.draw()


class AudioSettingsTab(QWidget):
    def __init__(self, go_back):
        super().__init__()
        self.go_back = go_back
        self.canvas = MatplotlibCanvas()
        layout = QVBoxLayout()

        self.device_selector = QComboBox()
        self.devices = AudioDevices()
        self.device_map = {}

        input_devices_info = self.devices.get_hostapi_with_devices(kind='input')
        for api_name, devices in input_devices_info.items():
            for dev_id, dev_info in devices.items():
                name = dev_info["name"]
                self.device_selector.addItem(name)
                self.device_map[name] = int(dev_id)

        layout.addWidget(QLabel("Выберите микрофон:"))
        layout.addWidget(self.device_selector)

        self.threshold_input = QLineEdit()
        layout.addWidget(QLabel("Порог слышимости (дБ):"))
        layout.addWidget(self.threshold_input)

        test_btn = QPushButton("Узнать (5 сек)")
        test_btn.clicked.connect(self.start_measurement)
        layout.addWidget(test_btn)
        layout.addWidget(self.canvas)

        btns = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        back_btn = QPushButton("Назад")
        save_btn.clicked.connect(self.save)
        back_btn.clicked.connect(self.go_back)
        btns.addWidget(save_btn)
        btns.addWidget(back_btn)
        layout.addLayout(btns)

        self.setLayout(layout)

    def load(self, config):
        mic_index = config.get("mic_device_index")
        if mic_index is not None:
            for name, index in self.device_map.items():
                if index == mic_index:
                    self.device_selector.setCurrentText(name)
                    break
        self.threshold_input.setText(str(config.get("silence_threshold_db", 50)))

    def save(self):
        config = load_config()
        selected_name = self.device_selector.currentText()
        config["mic_device_index"] = self.device_map.get(selected_name)
        config["silence_threshold_db"] = float(self.threshold_input.text())
        save_config(config)

    def start_measurement(self):
        self.analyzer = IntensityAnalyzer()
        self.analyzer.live_update.connect(self.canvas.plot_live)
        self.analyzer.result_ready.connect(lambda avg, t, i: self.threshold_input.setText(str(round(avg))))
        self.analyzer.start()


class TaskSettingsTab(QWidget):
    def __init__(self, go_back, on_save):
        super().__init__()
        self.go_back = go_back
        self.on_save = on_save
        layout = QVBoxLayout()
        layout.setSpacing(8)

        self.name_input = QLineEdit()
        layout.addWidget(QLabel("Название задания:"))
        layout.addWidget(self.name_input)

        self.duration_input = QLineEdit()
        layout.addWidget(QLabel("Время непрерывного говорения (сек):"))
        layout.addWidget(self.duration_input)

        self.text_input = QLineEdit()
        layout.addWidget(QLabel("Текст задания:"))
        layout.addWidget(self.text_input)

        btns = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        back_btn = QPushButton("Назад")
        save_btn.clicked.connect(self.save)
        back_btn.clicked.connect(self.go_back)
        btns.addWidget(save_btn)
        btns.addWidget(back_btn)
        layout.addLayout(btns)

        self.setLayout(layout)
        self.current_task = None

    def load_new(self):
        self.name_input.setText("")
        self.duration_input.setText("2")
        self.text_input.setText("МА")
        self.current_task = None

    def load(self, task):
        self.name_input.setText(task.get("name", ""))
        self.duration_input.setText(str(task.get("duration", 2)))
        self.text_input.setText(task.get("text", "МА"))
        self.current_task = task

    def save(self):
        name = self.name_input.text()
        duration = float(self.duration_input.text())
        text = self.text_input.text()

        tasks = load_tasks()
        if self.current_task:
            # обновляем
            for t in tasks:
                if t.get("name") == self.current_task.get("name"):
                    t["name"] = name
                    t["duration"] = duration
                    t["text"] = text
                    break
        else:
            # добавляем новое
            tasks.append({"name": name, "duration": duration, "text": text})

        save_tasks(tasks)
        if self.on_save:
            self.on_save()


class MainTab(QWidget):
    def __init__(self, stacked_widget, audio_tab, task_tab, task_editor):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.audio_tab = audio_tab
        self.task_tab = task_tab
        self.task_editor = task_editor

        layout = QVBoxLayout()

        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self.edit_task)
        layout.addWidget(QLabel("Выберите задание:"))
        layout.addWidget(self.task_list)

        btns_layout = QHBoxLayout()
        audio_btn = QPushButton("Настройка аудио")
        add_btn = QPushButton("Добавить задание")
        start_btn = QPushButton("Старт игры")

        audio_btn.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(audio_tab))
        add_btn.clicked.connect(self.add_task)
        start_btn.clicked.connect(self.start_game)

        btns_layout.addWidget(audio_btn)
        btns_layout.addWidget(add_btn)
        btns_layout.addWidget(start_btn)
        layout.addLayout(btns_layout)

        self.setLayout(layout)
        self.update_task_list()

    def update_task_list(self):
        self.task_list.clear()
        for task in load_tasks():
            self.task_list.addItem(task["name"])

    def add_task(self):
        self.task_editor.load_new()
        self.stacked_widget.setCurrentWidget(self.task_tab)

    def edit_task(self, item):
        name = item.text()
        for task in load_tasks():
            if task["name"] == name:
                self.task_editor.load(task)
                self.stacked_widget.setCurrentWidget(self.task_tab)
                break

    def start_game(self):
        current = self.task_list.currentItem()
        if not current:
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return

        name = current.text()
        for task in load_tasks():
            if task["name"] == name:
                print("Старт игры с заданием:", task)
                self.close()  # закрываем GUI перед стартом игры (по желанию)
                game = VoiceBowlingGame(task=task)
                game.setup()
                arcade.run()
                break

class BowlingConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Настройки голосового боулинга")
        self.setMinimumWidth(700)

        self.stacked = QStackedWidget()
        self.task_editor = TaskSettingsTab(go_back=self.show_main, on_save=self.on_task_save)
        self.audio_tab = AudioSettingsTab(go_back=self.show_main)
        self.main_tab = MainTab(self.stacked, self.audio_tab, self.task_editor, self.task_editor)

        self.stacked.addWidget(self.main_tab)
        self.stacked.addWidget(self.audio_tab)
        self.stacked.addWidget(self.task_editor)

        layout = QVBoxLayout()
        layout.addWidget(self.stacked)
        self.setLayout(layout)

    def show_main(self):
        self.main_tab.update_task_list()
        self.stacked.setCurrentWidget(self.main_tab)

    def on_task_save(self):
        self.show_main()


if __name__ == '__main__':
    import parselmouth

    app = QApplication(sys.argv)
    window = BowlingConfigWindow()
    window.show()
    sys.exit(app.exec())
