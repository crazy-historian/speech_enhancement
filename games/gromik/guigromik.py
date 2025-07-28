import sys
import json
import time
import numpy as np
import parselmouth
from pathlib import Path
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from audiochains.devices import AudioDevices
from PyQt6.QtWidgets import QComboBox

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QHBoxLayout, QCheckBox, QListWidget, QMessageBox, QGridLayout, QDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

TASKS_FILE = Path("games/gromik/components/tasks_gromik.json")
CONFIG_FILE = Path("games/gromik/components/config_gromik.json")

def load_config():
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    return {}

def save_config(data):
    CONFIG_FILE.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")

BLOCKSIZE = 1024
RECORD_DURATION = 5

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
        self.fig = Figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.ax.set_title("Интенсивность голоса")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Интенсивность (dB)")
        self.ax.grid(True)

    def plot(self, times, intensities):
        self.ax.clear()
        self.ax.plot(times, intensities, label="Интенсивность")
        self.ax.set_title("Интенсивность голоса")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Интенсивность (dB)")
        self.ax.grid(True)
        self.ax.legend()
        self.draw()
    def clear_plot(self):
        self.ax.clear()
        self.ax.set_title("Интенсивность голоса")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Интенсивность (dB)")
        self.ax.grid(True)
        self.draw()

    def plot_live(self, times, intensities):
        if not hasattr(self, 'fig') or not self.fig:
            return  # предотвращаем обращение к удалённому объекту
        self.ax.clear()
        self.ax.plot(times, intensities, label="Интенсивность")
        self.ax.set_title("Интенсивность голоса")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Интенсивность (dB)")
        self.ax.grid(True)
        self.ax.legend()
        self.draw()

class TaskEditor(QWidget):
    def __init__(self, task=None, on_save=None):
        super().__init__()
        self.setWindowTitle("Редактор задания")
        self.setMinimumWidth(1000)
        self.task = task or {}
        self.on_save = on_save
        self.canvas = MatplotlibCanvas()
        self.current_line = None
        self.times = []
        self.intensities = []
        #self.analyzer.live_update.connect(self.plot_live)

        main_layout = QHBoxLayout()
        form_layout = QVBoxLayout()

        def add_labeled_input(label, key, default=""):
            container = QHBoxLayout()
            line = QLineEdit()
            line.setText(str(self.task.get(key, default)))
            btn = QPushButton("Узнать")
            container.addWidget(QLabel(label))
            container.addWidget(line)
            container.addWidget(btn)
            form_layout.addLayout(container)
            return line, btn

        self.name = QLineEdit(self.task.get("name", ""))
        form_layout.addWidget(QLabel("Название задания"))
        form_layout.addWidget(self.name)

        self.quiet, btn_quiet = add_labeled_input("Тихий голос (дБ)", "quiet")
        self.norm, btn_norm = add_labeled_input("Нормальный голос (дБ)", "norm")
        self.loud, btn_loud = add_labeled_input("Громкий голос (дБ)", "loud")

        self.intensity_buttons = [(btn_quiet, self.quiet), (btn_norm, self.norm), (btn_loud, self.loud)]
        for button, line in self.intensity_buttons:
            button.clicked.connect(lambda _, l=line: self.measure_intensity(l))

        self.gen_quiet = QCheckBox("Генерировать тихие задания")
        self.gen_quiet.setChecked(self.task.get("gen_quiet", False))
        form_layout.addWidget(self.gen_quiet)

        self.gen_norm = QCheckBox("Генерировать нормальные задания")
        self.gen_norm.setChecked(self.task.get("gen_norm", True))
        form_layout.addWidget(self.gen_norm)

        self.gen_loud = QCheckBox("Генерировать громкие задания")
        self.gen_loud.setChecked(self.task.get("gen_loud", True))
        form_layout.addWidget(self.gen_loud)

        self.frequency = QLineEdit(str(self.task.get("frequency", "6")))
        form_layout.addWidget(QLabel("Частота появления заданий (сек)"))
        form_layout.addWidget(self.frequency)

        self.duration = QLineEdit(str(self.task.get("duration", "60")))
        form_layout.addWidget(QLabel("Длительность сессии (сек)"))
        form_layout.addWidget(self.duration)

        self.artifacts_count = QLineEdit(str(self.task.get("artifacts_count", "5")))
        form_layout.addWidget(QLabel("Кол-во артефактов (в слоге, слове)"))
        form_layout.addWidget(self.artifacts_count)

        self.artifact_interval = QLineEdit(str(self.task.get("artifact_interval", "0.2")))
        form_layout.addWidget(QLabel("Расстояние между артефактами (сек)"))
        form_layout.addWidget(self.artifact_interval)

        self.text = QLineEdit(self.task.get("text", "МА"))
        form_layout.addWidget(QLabel("Текст задания"))
        form_layout.addWidget(self.text)

        self.smooth = QCheckBox("Сглаженное управление (слитно)")
        self.smooth.setChecked(self.task.get("smooth", True))
        form_layout.addWidget(self.smooth)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self.save_task)
        back_btn = QPushButton("Назад")
        back_btn.clicked.connect(self.close)
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(back_btn)
        form_layout.addLayout(btn_layout)

        # график справа
        

        main_layout.addLayout(form_layout, 2)
        main_layout.addWidget(self.canvas, 3)
        self.setLayout(main_layout)

    def measure_intensity(self, target_line):
        self.times = []
        self.intensities = []
        self.canvas.clear_plot()

        self.analyzer = IntensityAnalyzer()
        self.analyzer.live_update.connect(self.plot_live)
        self.analyzer.result_ready.connect(lambda avg, t, i: self.handle_result(avg, t, i, target_line))
        self.analyzer.start()
    def plot_live(self, times, intensities):
        if hasattr(self, 'canvas') and self.canvas:  # простая проверка
            self.canvas.plot(times, intensities)
    
    def closeEvent(self, event):
        if hasattr(self, 'analyzer') and self.analyzer.isRunning():
            self.analyzer.stop()
            self.analyzer.wait()
        super().closeEvent(event)
    
    def update_graph(self, t, i):
        self.times.append(t)
        self.intensities.append(i)
        self.canvas.plot_live(self.times, self.intensities)
    
    def finish_measurement(self, avg, target_line):
        target_line.setText(str(round(avg)))

    def handle_result(self, avg, times, intensities, target_line):
        target_line.setText(str(round(avg)))
        self.canvas.plot(times, intensities)

    def save_task(self):
        new_task = {
            "name": self.name.text(),
            "quiet": int(self.quiet.text()),
            "norm": int(self.norm.text()),
            "loud": int(self.loud.text()),
            "gen_quiet": self.gen_quiet.isChecked(),
            "gen_norm": self.gen_norm.isChecked(),
            "gen_loud": self.gen_loud.isChecked(),
            "frequency": float(self.frequency.text()),
            "duration": int(self.duration.text()),
            "artifacts_count": int(self.artifacts_count.text()),
            "artifact_interval": float(self.artifact_interval.text()),
            "text": self.text.text(),
            "smooth": self.smooth.isChecked()
        }

        tasks = []
        if TASKS_FILE.exists():
            tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))

        found = False
        for i, t in enumerate(tasks):
            if t.get("name") == new_task["name"]:
                tasks[i] = new_task
                found = True
                break
        if not found:
            tasks.append(new_task)

        TASKS_FILE.write_text(json.dumps(tasks, indent=4, ensure_ascii=False), encoding="utf-8")
        if self.on_save:
            self.on_save()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Задания")
        layout = QVBoxLayout()
        self.device_selector = QComboBox()
        self.devices = AudioDevices()
        self.device_map = {}

        input_devices_info = self.devices.get_hostapi_with_devices(kind='input')
        input_devices = []

        for api_name, devices in input_devices_info.items():
            for dev_id, dev_info in devices.items():
                input_devices.append({
                    "name": dev_info["name"],
                    "index": int(dev_id)
                })
        for dev in input_devices:
            self.device_selector.addItem(dev["name"])
            self.device_map[dev["name"]] = dev["index"]

        layout.addWidget(QLabel("Выберите микрофон"))
        layout.addWidget(self.device_selector)

        # загрузка сохраненного устройства
        config = load_config()
        saved_index = config.get("mic_device_index")
        if saved_index is not None:
            for name, index in self.device_map.items():
                if index == saved_index:
                    self.device_selector.setCurrentText(name)
                    break

        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self.edit_task)
        layout.addWidget(self.task_list)

        add_btn = QPushButton("Добавить задание")
        add_btn.clicked.connect(self.add_task)
        layout.addWidget(add_btn)

        start_btn = QPushButton("Старт игры")
        start_btn.clicked.connect(self.start_game)
        layout.addWidget(start_btn)

        self.setLayout(layout)
        self.load_tasks()

    def load_tasks(self):
        self.task_list.clear()
        if TASKS_FILE.exists():
            tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
            for task in tasks:
                self.task_list.addItem(task["name"])

    def get_selected_task(self):
        task = None
        selected_name = self.device_selector.currentText()
        selected_index = self.device_map.get(selected_name)
        save_config({"mic_device_index": selected_index})

        selected = self.task_list.currentItem()
        if selected:
            name = selected.text()
            tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
            for t in tasks:
                if t["name"] == name:
                    task = t
        return task

    def edit_task(self):
        task = self.get_selected_task()
        if task:
            editor = TaskEditor(task, on_save=self.load_tasks)
            editor.show()

    def add_task(self):
        editor = TaskEditor(on_save=self.load_tasks)
        editor.show()

    def start_game(self):
        task = self.get_selected_task()
        if not task:
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return
        selected_name = self.device_selector.currentText()
        selected_index = self.device_map.get(selected_name)
        save_config({"mic_device_index": selected_index})
        print("Старт игры с заданием:", task)
    
        

def select_task_gui():
    import sys
    from PyQt6.QtWidgets import QListWidget, QPushButton, QVBoxLayout, QDialog, QDialogButtonBox
    from PyQt6.QtCore import QCoreApplication
    import json
    from pathlib import Path

    TASKS_FILE = Path("games/gromik/components/tasks_gromik.json")

    class TaskSelector(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Выбор задания")
            self.setMinimumWidth(400)
            layout = QVBoxLayout()

            self.task_list = QListWidget()
            layout.addWidget(self.task_list)

            self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            layout.addWidget(self.button_box)

            self.setLayout(layout)

            self.tasks = []
            if TASKS_FILE.exists():
                self.tasks = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
                for task in self.tasks:
                    self.task_list.addItem(task["name"])

            self.button_box.accepted.connect(self.accept)
            self.button_box.rejected.connect(self.reject)

        def get_selected_task(self):
            selected = self.task_list.currentItem()
            if selected:
                name = selected.text()
                for task in self.tasks:
                    if task["name"] == name:
                        return task
            return None

    # 🔥 вот тут создаём QApplication, если ещё не создан
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = TaskSelector()
    result = dialog.exec()

    if result == QDialog.DialogCode.Accepted:
        return dialog.get_selected_task()
    return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
