from typing import Optional
from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QLabel,
    QTextEdit, QComboBox, QDoubleSpinBox, QMessageBox
)
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import parselmouth
import time

from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

from .profiles import get_audio, upsert_task, get_settings

# ---- параметры анализа ----
BLOCKSIZE = 1024
RECORD_SEC = 5
PITCH_FLOOR = 80
PITCH_CEILING = 600
VOICING_THRESHOLD = 0.6

# ---- поток замера питча ----
class PitchAnalyzer(QThread):
    live_update = pyqtSignal(list, list)
    result_ready = pyqtSignal(float, list, list)

    def __init__(self, device_index=None, silence_db: float = 45.0, blocks_to_silent: int = 2):
        super().__init__()
        self.device_index = device_index
        self._running = True
        self.silence_db = silence_db
        self.blocks_to_silent = blocks_to_silent

    def stop(self):
        self._running = False
        self.requestInterruption()

    def run(self):
        try:
            with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2,
                             device=self.device_index) as stream:
                stream.set_methods(UnpackRawInFloat32())
                start = time.time()
                t, p = [], []
                cur = 0.0
                silent = 0
                last_pitch = None

                while (time.time() - start) < RECORD_SEC and self._running and not self.isInterruptionRequested():
                    raw = stream.read(BLOCKSIZE)
                    if not raw:
                        continue
                    sig = stream.chain_of_methods(raw)
                    snd = parselmouth.Sound(values=sig, sampling_frequency=stream.samplerate)
                    intensity = snd.to_intensity().values.T.flatten()
                    avg_db = float(np.mean(intensity)) if intensity.size else -50.0

                    pitch = snd.to_pitch_ac(
                        time_step=0.01, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                        voicing_threshold=VOICING_THRESHOLD
                    ).selected_array['frequency']
                    pitch[(pitch == 0) | (pitch > PITCH_CEILING)] = np.nan
                    avg_pitch = float(np.nanmean(pitch)) if np.any(~np.isnan(pitch)) else None

                    if avg_db > self.silence_db:
                        silent = 0
                        last_pitch = avg_pitch
                    else:
                        silent += 1
                        if silent >= self.blocks_to_silent:
                            last_pitch = None

                    t.append(cur); p.append(last_pitch); cur += BLOCKSIZE / 16000.0
                    self.live_update.emit(t.copy(), p.copy())

                clean = [x for x in p if x is not None]
                overall = float(np.mean(clean)) if clean else 0.0
                self.result_ready.emit(overall, t, p)
        except Exception as e:
            print(f"[PitchAnalyzer] {e}")

# ---- канвас ----
class MatplotlibCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self._setup()

    def _setup(self):
        self.ax.set_title("Pitch (Hz)")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Hz")
        self.ax.set_ylim(PITCH_FLOOR, PITCH_CEILING)
        self.ax.grid(True)

    def plot(self, t, p):
        self.ax.clear(); self._setup()
        self.ax.plot(t, [x if x else 0 for x in p], label="Pitch")
        self.ax.legend(); self.draw()

# ---- редактор задания VoiceCar ----
class TaskEditor(QWidget):
    """
    Поля voice_car:
      - name: str
      - frequency: "low" | "normal" | "high" | [min, max]
      - duration: float (сек)
      - max_out_of_zone: float (сек)
      - description: str
    + живой график питча и кнопка замера
    """
    def __init__(self, task: Optional[dict] = None, on_save=None, on_close=None, profile_name: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Редактор задания (VoiceCar)")
        self.setMinimumWidth(980)
        self.profile_name = profile_name
        self.task = task or {}
        self.on_save = on_save
        self.on_close = on_close
        self._saved = False

        self.canvas = MatplotlibCanvas()
        self.an = None

        # настройки профиля (для low/normal/high)
        self.settings = get_settings(self.profile_name) if self.profile_name else {
            "pitch_ranges": {"low": [150,199], "normal": [200,250], "high":[251,300]},
            "blocks_to_silent": 2
        }

        # ---- левая панель формы ----
        form = QVBoxLayout()

        self.name_edit = QLineEdit(self.task.get("name", ""))
        form.addWidget(QLabel("Название задания:"))
        form.addWidget(self.name_edit)

        # частотная зона
        pr = self.settings.get("pitch_ranges", {})
        self.freq_combo = QComboBox()
        self.freq_combo.addItem(f"low ({pr.get('low',[0,0])[0]}–{pr.get('low',[0,0])[1]})")
        self.freq_combo.addItem(f"normal ({pr.get('normal',[0,0])[0]}–{pr.get('normal',[0,0])[1]})")
        self.freq_combo.addItem(f"high ({pr.get('high',[0,0])[0]}–{pr.get('high',[0,0])[1]})")
        self.freq_combo.addItem("custom")
        form.addWidget(QLabel("Частотная зона:"))
        form.addWidget(self.freq_combo)

        self.custom_freq = QLineEdit()
        self.custom_freq.setPlaceholderText("например: 180-220")
        form.addWidget(self.custom_freq)

        # длительность и выход за пределы
        self.duration_box = QDoubleSpinBox(); self.duration_box.setDecimals(1); self.duration_box.setRange(0.1, 3600.0)
        self.duration_box.setValue(float(self.task.get("duration",  self.settings.get("base_duration", 4.0))))
        form.addWidget(QLabel("Длительность (сек):")); form.addWidget(self.duration_box)

        self.max_out_box = QDoubleSpinBox(); self.max_out_box.setDecimals(2); self.max_out_box.setRange(0.0, 60.0)
        self.max_out_box.setValue(float(self.task.get("max_out_of_zone", self.settings.get("max_out_of_zone", 0.5))))
        form.addWidget(QLabel("Макс. пребывание за пределами (сек):")); form.addWidget(self.max_out_box)

        self.desc = QTextEdit(self.task.get("description", ""))
        form.addWidget(QLabel("Описание:")); form.addWidget(self.desc)

        # замер
        self.measure_btn = QPushButton("Замерить питч (5 сек)")
        self.measure_btn.clicked.connect(self.measure_pitch)
        self.avg_label = QLabel("Средний: — Hz")
        form.addWidget(self.measure_btn); form.addWidget(self.avg_label)

        # кнопки
        btns = QHBoxLayout()
        save_btn = QPushButton("Сохранить")
        back_btn = QPushButton("Назад")
        save_btn.clicked.connect(self.save_task)
        back_btn.clicked.connect(self.close)
        btns.addWidget(save_btn); btns.addWidget(back_btn)
        form.addLayout(btns)

        # правый график
        root = QHBoxLayout(self)
        root.addLayout(form, 1)
        root.addWidget(self.canvas, 2)
        self.setLayout(root)

        # загрузка значения частоты
        self._init_frequency_fields()

    def _init_frequency_fields(self):
        # выставить комбо и кастом при редактировании
        freq = self.task.get("frequency", "normal")
        if isinstance(freq, list) and len(freq) == 2:
            self.freq_combo.setCurrentText("custom")
            self.custom_freq.setText(f"{int(freq[0])}-{int(freq[1])}")
        else:
            # freq ожидается "low"/"normal"/"high"
            # выставляем совпадающий префикс элемента
            key = str(freq)
            for i in range(self.freq_combo.count()):
                if self.freq_combo.itemText(i).startswith(key):
                    self.freq_combo.setCurrentIndex(i)
                    break

        # показывать/скрывать поле custom
        def toggle(_txt):
            self.custom_freq.setVisible("custom" in _txt)
        self.freq_combo.currentTextChanged.connect(toggle)
        toggle(self.freq_combo.currentText())

    # ---- измерение ----
    def measure_pitch(self):
        if self.an and self.an.isRunning():
            self.an.stop(); self.an.wait()

        audio = get_audio(self.profile_name) if self.profile_name else {"mic_device_index": None, "silence_threshold_db": 45.0}
        device = audio.get("mic_device_index")
        silence_db = float(audio.get("silence_threshold_db", 45.0))
        blocks_to_silent = int(self.settings.get("blocks_to_silent", 2))

        self.an = PitchAnalyzer(device_index=device, silence_db=silence_db, blocks_to_silent=blocks_to_silent)
        self.an.live_update.connect(self.canvas.plot, QtCore.Qt.ConnectionType.QueuedConnection)
        self.an.result_ready.connect(self._on_measure_done, QtCore.Qt.ConnectionType.QueuedConnection)
        self.an.finished.connect(lambda: setattr(self, "an", None))
        self.canvas.plot([], [])
        self.an.start()

    def _on_measure_done(self, avg_pitch: float, t, p):
        self.avg_label.setText(f"Средний: {round(avg_pitch)} Hz")
        self.canvas.plot(t, p)
        # если выбрана custom и поле пусто — подставим узкий диапазон вокруг среднего
        if "custom" in self.freq_combo.currentText() and not self.custom_freq.text().strip():
            lo = max(PITCH_FLOOR, int(avg_pitch - 15))
            hi = min(PITCH_CEILING, int(avg_pitch + 15))
            self.custom_freq.setText(f"{lo}-{hi}")

    # ---- сохранение ----
    def save_task(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Ошибка", "Введите название задания"); return

        freq_text = self.freq_combo.currentText()
        if "custom" in freq_text:
            txt = self.custom_freq.text().strip()
            try:
                a, b = txt.split("-")
                frequency = [int(a), int(b)]
            except Exception:
                QMessageBox.warning(self, "Ошибка", "Неверный формат диапазона. Пример: 180-220")
                return
        else:
            frequency = freq_text.split()[0]  # "low"/"normal"/"high"

        new_task = {
            "name": name,
            "frequency": frequency,
            "duration": float(self.duration_box.value()),
            "max_out_of_zone": float(self.max_out_box.value()),
            "description": self.desc.toPlainText().strip()
        }

        if self.profile_name:
            upsert_task(self.profile_name, new_task)

        self._saved = True
        if callable(self.on_save):
            QTimer.singleShot(0, self.on_save)
        self.close()

    # ---- выход ----
    def closeEvent(self, e):
        if self.an:
            try:
                self.an.stop()
            except Exception:
                pass
            if self.an.isRunning():
                self.an.wait(800)
            self.an = None
        if not self._saved and callable(self.on_close):
            QTimer.singleShot(0, self.on_close)
        super().closeEvent(e)
