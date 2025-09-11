from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QLabel, QComboBox

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from audiochains.devices import AudioDevices
from .analyzers import IntensityAnalyzer
from .profiles import get_audio, set_audio


class MatplotlibCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 2))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self._setup()

    def _setup(self):
        self.ax.set_title("Интенсивность голоса (dB)")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("dB")
        self.ax.grid(True)

    def plot_live(self, times, intensities):
        self.ax.clear(); self._setup()
        self.ax.plot(times, intensities, label="Интенсивность")
        self.ax.legend(); self.draw()


class AudioSettingsWindow(QWidget):
    def __init__(self, profile_name: str, go_back=None):
        super().__init__()
        self.profile_name = profile_name
        self.go_back = go_back
        self.setWindowTitle(f"Настройки аудио — {self.profile_name}")
        self.canvas = MatplotlibCanvas()

        layout = QVBoxLayout(self)
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
        back_btn.clicked.connect(self._back)
        btns.addWidget(save_btn)
        btns.addWidget(back_btn)
        layout.addLayout(btns)

        self._an = None
        self._load()

    def _load(self):
        audio = get_audio(self.profile_name)
        mic_index = audio.get("mic_device_index")
        if mic_index is not None:
            for name, index in self.device_map.items():
                if index == mic_index:
                    self.device_selector.setCurrentText(name)
                    break
        self.threshold_input.setText(str(audio.get("silence_threshold_db", 50)))

    def save(self):
        selected_name = self.device_selector.currentText()
        mic_idx = self.device_map.get(selected_name)
        try:
            th = float(self.threshold_input.text())
        except ValueError:
            th = 50.0
        set_audio(self.profile_name, mic_device_index=mic_idx, silence_threshold_db=th)

    def start_measurement(self):
        if self._an and self._an.isRunning():
            self._an.stop(); self._an.wait()
        selected_name = self.device_selector.currentText()
        device_idx = self.device_map.get(selected_name)
        self._an = IntensityAnalyzer(device_index=device_idx)
        self._an.live_update.connect(self.canvas.plot_live)
        self._an.result_ready.connect(lambda avg, t, i: self.threshold_input.setText(str(round(avg))))
        self._an.start()

    def _back(self):
        self.close()
        if self.go_back:
            self.go_back()

    def closeEvent(self, e):
        if self._an and self._an.isRunning():
            self._an.stop(); self._an.wait()
        return super().closeEvent(e)