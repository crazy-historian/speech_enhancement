import sys
import json
import time
import numpy as np
import parselmouth
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit,
    QHBoxLayout, QListWidget, QStackedWidget, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from audiochains.devices import AudioDevices
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

CONFIG_FILE = Path("profiles/syllable_config.json")

# ----------------------------- Работа с JSON -----------------------------
def load_config():
    if CONFIG_FILE.exists() and CONFIG_FILE.read_text().strip():
        return json.loads(CONFIG_FILE.read_text())
    return {"audio": {}, "tasks": []}

def save_config(data):
    CONFIG_FILE.write_text(json.dumps(data, indent=4, ensure_ascii=False))

def load_tasks():
    config = load_config()
    return config.get("tasks", [])

def save_tasks(tasks):
    config = load_config()
    config["tasks"] = tasks
    save_config(config)

# ----------------------------- Анализатор интенсивности -----------------------------
BLOCKSIZE = 1024
RECORD_DURATION = 5
PITCH_FLOOR = 100
PITCH_CEILING = 600
VOICING_THRESHOLD = 0.6
SILENCE_THRESHOLD_DB = 45.0
BLOCKS_TO_SILENT = 1

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
                import parselmouth
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

# ----------------------------- Анализатор слитности -----------------------------
class VoiceFlowAnalyzer(QThread):
    update_plot = pyqtSignal(list, list, list)
    average_pause_ready = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
            stream.set_methods(UnpackRawInFloat32())
            times, voice_states = [], []
            start_time = time.time()
            last_voice_state = 0
            silent_counter = 0

            tracking_pauses = False
            pause_start_time = None
            pause_durations = []
            pause_annotations = []

            while time.time() - start_time < RECORD_DURATION and self._running:
                raw_data = stream.read(BLOCKSIZE)
                if not raw_data:
                    continue

                signal = stream.chain_of_methods(raw_data)
                sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

                intensity_obj = sound.to_intensity()
                intensity_values = intensity_obj.values.T.flatten()
                avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

                pitch_obj = sound.to_pitch_ac(
                    time_step=0.01,
                    pitch_floor=PITCH_FLOOR,
                    pitch_ceiling=PITCH_CEILING,
                    voicing_threshold=VOICING_THRESHOLD
                )
                pitch_values = pitch_obj.selected_array['frequency']
                pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
                avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else None

                above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
                valid_pitch = (avg_pitch is not None and avg_pitch > PITCH_FLOOR)
                voice_state = 1 if above_silence_threshold and valid_pitch else 0

                if voice_state == 1:
                    silent_counter = 0
                else:
                    silent_counter += 1
                    if silent_counter < BLOCKS_TO_SILENT:
                        voice_state = last_voice_state
                    else:
                        voice_state = 0

                current_time = time.time() - start_time

                if not tracking_pauses and voice_state == 1:
                    tracking_pauses = True

                if tracking_pauses:
                    if last_voice_state == 1 and voice_state == 0:
                        pause_start_time = current_time
                    elif last_voice_state == 0 and voice_state == 1 and pause_start_time is not None:
                        pause_duration = current_time - pause_start_time
                        pause_durations.append(pause_duration)
                        pause_annotations.append((pause_start_time, pause_duration))
                        pause_start_time = None

                last_voice_state = voice_state

                times.append(current_time)
                voice_states.append(voice_state)
                self.update_plot.emit(times.copy(), voice_states.copy(), pause_annotations.copy())

            if pause_durations:
                avg_pause = sum(pause_durations) / len(pause_durations)
                self.average_pause_ready.emit(avg_pause)

# ----------------------------- Виджет графика -----------------------------
class FlowPlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 2))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setup_plot()

    def setup_plot(self):
        self.ax.set_title("Анализ слитности и раздельности речи")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Голос (0 / 1)")
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid(True)

    def update_graph(self, times, values, pauses):
        self.ax.clear()
        self.setup_plot()
        self.ax.plot(times, values, color="blue")

        for pause_start, pause_duration in pauses:
            self.ax.axvspan(pause_start, pause_start + pause_duration, color='orange', alpha=0.3)
            self.ax.text(pause_start + pause_duration / 2, 1.05, f"{pause_duration:.2f}s", ha="center", fontsize=8, color="darkred")

        self.draw()

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
        mic_index = config.get("audio", {}).get("mic_device_index")
        if mic_index is not None:
            for name, index in self.device_map.items():
                if index == mic_index:
                    self.device_selector.setCurrentText(name)
                    break
        self.threshold_input.setText(str(config.get("audio", {}).get("silence_threshold_db", 50)))

    def save(self):
        config = load_config()
        selected_name = self.device_selector.currentText()
        config["audio"] = {
            "mic_device_index": self.device_map.get(selected_name),
            "silence_threshold_db": float(self.threshold_input.text())
        }
        save_config(config)

    def start_measurement(self):
        self.analyzer = IntensityAnalyzer()
        self.analyzer.live_update.connect(self.canvas.plot_live)
        self.analyzer.result_ready.connect(lambda avg, t, i: self.threshold_input.setText(str(round(avg))))
        self.analyzer.start()

# --------------------------- Окно настройки задания ---------------------------
class TaskSettingsTab(QWidget):
    def __init__(self, go_back, on_save):
        super().__init__()
        self.go_back = go_back
        self.on_save = on_save
        self.current_task = None

        self.canvas = FlowPlotCanvas()

        layout = QHBoxLayout()
        left = QVBoxLayout()

        self.name_input = QLineEdit()
        self.duration_input = QLineEdit()
        self.wave_interval_input = QLineEdit()
        self.wave_size_input = QLineEdit()
        self.syllable_interval_input = QLineEdit()
        self.text_input = QLineEdit()

        left.addWidget(QLabel("Название задания:"))
        left.addWidget(self.name_input)
        left.addWidget(QLabel("Длительность игры (сек):"))
        left.addWidget(self.duration_input)
        left.addWidget(QLabel("Интервал между волнами (сек):"))
        left.addWidget(self.wave_interval_input)
        left.addWidget(QLabel("Кол-во слогов в волне:"))
        left.addWidget(self.wave_size_input)
        left.addWidget(QLabel("Интервал между слогами (сек):"))
        left.addWidget(self.syllable_interval_input)

        test_btn = QPushButton("Узнать")
        test_btn.clicked.connect(self.start_measurement)
        left.addWidget(test_btn)

        left.addWidget(QLabel("Текст задания (слоги):"))
        left.addWidget(self.text_input)

        btns = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        back_btn = QPushButton("Назад")
        save_btn.clicked.connect(self.save)
        back_btn.clicked.connect(self.go_back)
        btns.addWidget(save_btn)
        btns.addWidget(back_btn)
        left.addLayout(btns)

        layout.addLayout(left)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def load_new(self):
        self.current_task = None
        self.name_input.setText("")
        self.duration_input.setText("60")
        self.wave_interval_input.setText("3")
        self.wave_size_input.setText("3")
        self.syllable_interval_input.setText("0.5")
        self.text_input.setText("МА")

    def load(self, task):
        self.current_task = task
        self.name_input.setText(task.get("name", ""))
        self.duration_input.setText(str(task.get("duration", 60)))
        self.wave_interval_input.setText(str(task.get("wave_interval", 3)))
        self.wave_size_input.setText(str(task.get("wave_size", 3)))
        self.syllable_interval_input.setText(str(task.get("syllable_interval", 0.5)))
        self.text_input.setText(task.get("text", "МА"))

    def save(self):
        new_task = {
            "name": self.name_input.text(),
            "duration": float(self.duration_input.text()),
            "wave_interval": float(self.wave_interval_input.text()),
            "wave_size": int(self.wave_size_input.text()),
            "syllable_interval": float(self.syllable_interval_input.text()),
            "text": self.text_input.text()
        }

        tasks = load_tasks()
        if self.current_task:
            for t in tasks:
                if t == self.current_task:
                    t.update(new_task)
                    break
        else:
            tasks.append(new_task)

        save_tasks(tasks)
        if self.on_save:
            self.on_save()

    def start_measurement(self):
        self.analyzer = VoiceFlowAnalyzer()
        self.analyzer.update_plot.connect(self.canvas.update_graph)
        self.analyzer.average_pause_ready.connect(self.set_average_syllable_interval)
        self.analyzer.start()

    def set_average_syllable_interval(self, avg):
        self.syllable_interval_input.setText(f"{round(avg, 2)}")

# --------------------------- Главное окно ---------------------------
class MainTab(QWidget):
    def __init__(self, stacked_widget, audio_tab, task_editor, parent_window):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.audio_tab = audio_tab
        self.task_editor = task_editor
        self._parent = parent_window

        layout = QVBoxLayout()

        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self.edit_task)
        layout.addWidget(QLabel("Выберите задание:"))
        layout.addWidget(self.task_list)

        btns = QHBoxLayout()
        audio_btn = QPushButton("Настройка аудио")
        add_btn = QPushButton("Добавить задание")
        start_btn = QPushButton("Старт игры")

        audio_btn.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(audio_tab))
        add_btn.clicked.connect(self.add_task)
        start_btn.clicked.connect(self.start_game)

        btns.addWidget(audio_btn)
        btns.addWidget(add_btn)
        btns.addWidget(start_btn)
        layout.addLayout(btns)

        self.setLayout(layout)
        self.update_task_list()

    def update_task_list(self):
        self.task_list.clear()
        for task in load_tasks():
            self.task_list.addItem(task.get("name", "Без названия"))

    def add_task(self):
        self.task_editor.load_new()
        self.stacked_widget.setCurrentWidget(self.task_editor)

    def edit_task(self, item):
        name = item.text()
        for task in load_tasks():
            if task.get("name") == name:
                self.task_editor.load(task)
                self.stacked_widget.setCurrentWidget(self.task_editor)
                break

    def start_game(self):
        current = self.task_list.currentItem()
        if not current:
            QMessageBox.warning(self, "Нет задания", "Выберите задание")
            return
        name = current.text()
        for task in load_tasks():
            if task.get("name") == name:
                self._parent.selected_task = task
                self._parent.should_start_game = True
                self._parent.close()
                break
        
        

    def get_selected_task(self):
        current = self.task_list.currentItem()
        if not current:
            return None
        name = current.text()
        for task in load_tasks():
            if task.get("name") == name:
                return task
        return None

# --------------------------- Главное окно приложения ---------------------------
class SyllableConfigWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Настройка игры на слоги")
        self.setMinimumWidth(700)

        self.stacked = QStackedWidget()
        self.audio_tab = AudioSettingsTab(go_back=self.show_main)
        self.task_editor = TaskSettingsTab(go_back=self.show_main, on_save=self.show_main)
        self.main_tab = MainTab(self.stacked, self.audio_tab, self.task_editor, self)

        self.stacked.addWidget(self.main_tab)
        self.stacked.addWidget(self.audio_tab)
        self.stacked.addWidget(self.task_editor)

        layout = QVBoxLayout()
        layout.addWidget(self.stacked)
        self.setLayout(layout)
        self.selected_task = None
        self.should_start_game = False

    def show_main(self):
        self.main_tab.update_task_list()
        self.stacked.setCurrentWidget(self.main_tab)
        self.audio_tab.load(load_config())
    def get_selected_task(self):
        return self.main_tab.get_selected_task()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SyllableConfigWindow()
    window.show()
    sys.exit(app.exec())
