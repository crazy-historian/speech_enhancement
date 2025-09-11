from typing import Optional

from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QLabel, QCheckBox
from PyQt6.QtGui import QIntValidator, QDoubleValidator

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import parselmouth
import time
import numpy as np

from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

from .profiles import get_audio, upsert_task


# ---------------------------------------------
# ПАРАМЕТРЫ АНАЛИЗА
# ---------------------------------------------
BLOCKSIZE = 1024
RECORD_DURATION = 5
PITCH_FLOOR = 100
PITCH_CEILING = 600
VOICING_THRESHOLD = 0.6
SILENCE_THRESHOLD_DB = 45.0
BLOCKS_TO_SILENT = 2


# ---------------------------------------------
# ПОТОК ДЛЯ АНАЛИЗА PITCH В РЕАЛЬНОМ ВРЕМЕНИ
# ---------------------------------------------
class PitchAnalyzer(QThread):
    result_ready = pyqtSignal(float, list, list)
    live_update = pyqtSignal(list, list)

    def __init__(self, device_index=None):
        super().__init__()
        self._running = True
        self.device_index = device_index

    def stop(self):
        self._running = False
        self.requestInterruption()

    def run(self):
        try:
            with InputStream(
                samplerate=16000,
                blocksize=BLOCKSIZE,
                channels=1,
                sampwidth=2,
                device=self.device_index
            ) as stream:
                stream.set_methods(UnpackRawInFloat32())

                start_time = time.time()
                times, pitches = [], []
                current_time = 0.0
                silent_counter = 0
                last_valid_pitch = None

                while (time.time() - start_time < RECORD_DURATION) and self._running and not self.isInterruptionRequested():
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

                    if avg_intensity > SILENCE_THRESHOLD_DB:
                        silent_counter = 0
                        last_valid_pitch = avg_pitch
                    else:
                        silent_counter += 1
                        if silent_counter >= BLOCKS_TO_SILENT:
                            last_valid_pitch = None

                    times.append(current_time)
                    pitches.append(last_valid_pitch)
                    current_time += BLOCKSIZE / 16000

                    if self._running:
                        self.live_update.emit(times.copy(), pitches.copy())

                clean_pitches = [p for p in pitches if p is not None]
                overall_avg = np.mean(clean_pitches) if clean_pitches else 0.0

                if self._running:
                    self.result_ready.emit(overall_avg, times, pitches)

        except Exception as e:
            print(f"[PitchAnalyzer] Exception: {e}")


# ---------------------------------------------
# КАНВАС ДЛЯ ОТОБРАЖЕНИЯ ГРАФИКА ЧАСТОТЫ
# ---------------------------------------------
class MatplotlibCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(9, 5))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.ax.set_title("Частота (Hz)")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Pitch (Hz)")
        self.ax.set_ylim(PITCH_FLOOR, PITCH_CEILING)
        self.ax.grid(True)

    def plot(self, times, pitches):
        self.ax.clear()
        self.ax.set_title("Частота (Hz)")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Pitch (Hz)")
        self.ax.set_ylim(PITCH_FLOOR, PITCH_CEILING)
        self.ax.grid(True)
        self.ax.plot(times, [p if p else 0 for p in pitches], label="Pitch")
        self.ax.legend()
        self.draw()

    def clear_plot(self):
        self.ax.clear()
        self.ax.set_title("Частота (Hz)")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("Pitch (Hz)")
        self.ax.set_ylim(PITCH_FLOOR, PITCH_CEILING)
        self.ax.grid(True)
        self.draw()


# ---------------------------------------------
# ОКНО РЕДАКТОРА ЗАДАНИЯ (TaskEditor)
# ---------------------------------------------
class TaskEditor(QWidget):
    """
    Живёт внутри QStackedWidget.
    on_save() — вызывается после сохранения.
    on_close() — вызывается при закрытии без сохранения (кнопка «Назад»/крестик).
    """
    def __init__(self, task=None, on_save=None, on_close=None, profile_name: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Редактор задания (Pitch)")
        self.setMinimumWidth(1000)

        self.profile_name = profile_name
        self.task = task or {}
        self.on_save = on_save
        self.on_close = on_close
        self._saved = False

        self.canvas = MatplotlibCanvas()
        self.analyzer = None
        self.current_line = None

        layout = QHBoxLayout()
        form_layout = QVBoxLayout()

        def add_labeled_input(label, key, default=""):
            row = QHBoxLayout()
            lbl = QLabel(label)
            line = QLineEdit(str(self.task.get(key, default)))
            btn = QPushButton("Узнать")
            row.addWidget(lbl)
            row.addWidget(line)
            row.addWidget(btn)
            form_layout.addLayout(row)
            return line, btn

        self.name_edit = QLineEdit(str(self.task.get("name", "")))
        form_layout.addWidget(QLabel("Название задания:"))
        form_layout.addWidget(self.name_edit)

        self.quiet_line, btn_quiet = add_labeled_input("Тихий (Hz):", "quiet", "120")
        self.norm_line,  btn_norm  = add_labeled_input("Нормальный (Hz):", "norm",  "180")
        self.loud_line,  btn_loud  = add_labeled_input("Громкий (Hz):",   "loud",  "240")

        self.quiet_line.setValidator(QIntValidator(50, 1000, self))
        self.norm_line.setValidator(QIntValidator(50, 1000, self))
        self.loud_line.setValidator(QIntValidator(50, 1000, self))

        self.chk_quiet = QCheckBox("Генерировать 'тихий' уровень")
        self.chk_quiet.setChecked(self.task.get("gen_quiet", False))
        form_layout.addWidget(self.chk_quiet)

        self.chk_norm = QCheckBox("Генерировать 'нормальный' уровень")
        self.chk_norm.setChecked(self.task.get("gen_norm", True))
        form_layout.addWidget(self.chk_norm)

        self.chk_loud = QCheckBox("Генерировать 'громкий' уровень")
        self.chk_loud.setChecked(self.task.get("gen_loud", True))
        form_layout.addWidget(self.chk_loud)

        self.frequency_line = QLineEdit(str(self.task.get("frequency", "6")))
        self.duration_line = QLineEdit(str(self.task.get("duration", "60")))
        self.artifacts_count_line = QLineEdit(str(self.task.get("artifacts_count", "5")))
        self.artifact_interval_line = QLineEdit(str(self.task.get("artifact_interval", "0.2")))
        self.text_line = QLineEdit(str(self.task.get("text", "ДА")))

        self.frequency_line.setValidator(QDoubleValidator(0.1, 60.0, 2, self))
        self.duration_line.setValidator(QIntValidator(5, 3600, self))
        self.artifacts_count_line.setValidator(QIntValidator(1, 1000, self))
        self.artifact_interval_line.setValidator(QDoubleValidator(0.01, 10.0, 2, self))

        form_layout.addWidget(QLabel("Частота появления заданий (с)"))
        form_layout.addWidget(self.frequency_line)
        form_layout.addWidget(QLabel("Длительность игры (с)"))
        form_layout.addWidget(self.duration_line)
        form_layout.addWidget(QLabel("Кол-во артефактов (в волне)"))
        form_layout.addWidget(self.artifacts_count_line)
        form_layout.addWidget(QLabel("Интервал между артефактами (с)"))
        form_layout.addWidget(self.artifact_interval_line)
        form_layout.addWidget(QLabel("Текст задания (слог)"))
        form_layout.addWidget(self.text_line)

        self.smooth_chk = QCheckBox("Сглаженное управление (слитно)")
        self.smooth_chk.setChecked(self.task.get("smooth", True))
        form_layout.addWidget(self.smooth_chk)

        btns_layout = QHBoxLayout()
        btn_save = QPushButton("Сохранить")
        btn_back = QPushButton("Назад")
        btns_layout.addWidget(btn_save)
        btns_layout.addWidget(btn_back)
        form_layout.addLayout(btns_layout)

        layout.addLayout(form_layout, 1)
        layout.addWidget(self.canvas, 2)
        self.setLayout(layout)

        # кнопки «Узнать»
        for button, line_edit in ((btn_quiet, self.quiet_line), (btn_norm, self.norm_line), (btn_loud, self.loud_line)):
            button.clicked.connect(lambda _, le=line_edit: self.measure_pitch(le))

        btn_save.clicked.connect(self.save_task)
        btn_back.clicked.connect(self.close)

    # --------- Замер питча ---------
    def measure_pitch(self, target_line):
        if self.analyzer is not None and self.analyzer.isRunning():
            self.analyzer.stop()
            self.analyzer.wait()

        if self.canvas:
            self.canvas.clear_plot()

        audio = get_audio(self.profile_name) if self.profile_name else {"mic_device_index": None}
        device_index = audio.get("mic_device_index", None)

        self.analyzer = PitchAnalyzer(device_index=device_index)
        self.analyzer.setParent(self)
        self.analyzer.live_update.connect(
            self.plot_live,
            QtCore.Qt.ConnectionType.QueuedConnection
        )
        self.analyzer.result_ready.connect(
            lambda avg_pitch, times, pitches: self.handle_result(avg_pitch, times, pitches, target_line),
            QtCore.Qt.ConnectionType.QueuedConnection
        )
        self.analyzer.finished.connect(lambda: setattr(self, "analyzer", None))
        self.analyzer.start()

    def plot_live(self, times, pitches):
        if not self.isVisible() or not hasattr(self, "canvas") or self.canvas is None:
            return
        try:
            self.canvas.plot(times, pitches)
        except RuntimeError:
            pass

    def handle_result(self, avg_pitch, times, pitches, target_line):
        if not self.isVisible() or not hasattr(self, "canvas") or self.canvas is None:
            return
        target_line.setText(str(round(avg_pitch)))
        try:
            self.canvas.plot(times, pitches)
        except RuntimeError:
            pass

    def closeEvent(self, event):
        # стопнём поток
        if self.analyzer is not None:
            try:
                self.analyzer.live_update.disconnect(self.plot_live)
            except TypeError:
                pass
            self.analyzer.blockSignals(True)
            try:
                self.analyzer.requestInterruption()
            except Exception:
                pass
            self.analyzer.stop()
            if self.analyzer.isRunning():
                self.analyzer.wait(1000)
            self.analyzer = None

        # вызовем колбэк "закрыто без сохранения" уже в следующем тике,
        # чтобы родитель успел обработать удаление дочерних виджетов
        if not self._saved and callable(self.on_close):
            cb = self.on_close
            QTimer.singleShot(0, lambda: cb())

        super().closeEvent(event)

    # --------- Сохранение ---------
    def save_task(self):
        def int_or(le: QLineEdit, default: int) -> int:
            txt = le.text().strip()
            return int(txt) if txt else default

        def float_or(le: QLineEdit, default: float) -> float:
            txt = le.text().strip()
            return float(txt) if txt else default

        name = self.name_edit.text().strip()
        new_task = {
            "name": name,
            "quiet": int_or(self.quiet_line, 120),
            "norm": int_or(self.norm_line, 180),
            "loud": int_or(self.loud_line, 240),
            "gen_quiet": self.chk_quiet.isChecked(),
            "gen_norm": self.chk_norm.isChecked(),
            "gen_loud": self.chk_loud.isChecked(),
            "frequency": float_or(self.frequency_line, 6.0),
            "duration": int_or(self.duration_line, 60),
            "artifacts_count": int_or(self.artifacts_count_line, 5),
            "artifact_interval": float_or(self.artifact_interval_line, 0.2),
            "text": self.text_line.text().strip(),
            "smooth": self.smooth_chk.isChecked()
        }

        if not self.profile_name:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Профиль не задан", "Редактор открыт без профиля")
        else:
            upsert_task(self.profile_name, new_task)

        self._saved = True
        if callable(self.on_save):
            try:
                # вызываем в следующем тике — безопаснее для родителя/стека
                QTimer.singleShot(0, self.on_save)
            except Exception:
                pass

        self.close()
