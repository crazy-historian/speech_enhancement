# games/gromik/gui/task_editor_intensity.py
from __future__ import annotations
from typing import Optional

from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QLabel, QCheckBox
from PyQt6.QtGui import QDoubleValidator, QIntValidator

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import parselmouth
import numpy as np
import time

from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

from .profiles import get_audio, upsert_task

# ---------------------------------------------
# ПАРАМЕТРЫ АНАЛИЗА
# ---------------------------------------------
BLOCKSIZE = 1024
RECORD_DURATION = 5.0            # секунд
EMA_ALPHA = 1                # сглаживание графика по экспоненте
INT_FALLBACK_DB = -50.0          # на случай пустого буфера

# ---------------------------------------------
# ПОТОК ДЛЯ АНАЛИЗА ИНТЕНСИВНОСТИ (dB) В РЕАЛЬНОМ ВРЕМЕНИ
# ---------------------------------------------
class IntensityAnalyzer(QThread):
    result_ready = pyqtSignal(float, list, list)   # avg_db, times, intensities_db
    live_update  = pyqtSignal(list, list)          # times, intensities_db

    def __init__(self, device_index: Optional[int] = None, silence_threshold_db: float = 45.0, ema_alpha: float = EMA_ALPHA):
        super().__init__()
        self._running = True
        self.device_index = device_index
        self.silence_threshold_db = float(silence_threshold_db)
        self.ema_alpha = float(ema_alpha)

    def stop(self):
        self._running = False
        try:
            self.requestInterruption()
        except Exception:
            pass

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
                times: list[float] = []
                intensities: list[float] = []
                cur_t = 0.0
                ema_prev = None

                while (time.time() - start_time) < RECORD_DURATION and self._running and not self.isInterruptionRequested():
                    raw = stream.read(BLOCKSIZE)
                    if not raw:
                        continue

                    sig = stream.chain_of_methods(raw)
                    sound = parselmouth.Sound(values=sig, sampling_frequency=stream.samplerate)

                    # dB относительная интенсивность (Praat)
                    iv = sound.to_intensity().values.T.flatten()
                    inst_db = float(np.mean(iv)) if iv.size else INT_FALLBACK_DB

                    # экспоненциальное сглаживание для живого графика
                    if ema_prev is None:
                        ema_prev = inst_db
                    else:
                        ema_prev = self.ema_alpha * inst_db + (1.0 - self.ema_alpha) * ema_prev

                    times.append(cur_t)
                    intensities.append(float(ema_prev))
                    cur_t += BLOCKSIZE / 16000.0

                    # онлайновый апдейт
                    if self._running:
                        self.live_update.emit(times.copy(), intensities.copy())

                # среднее по "озвученным" блокам (выше порога тишины).
                voiced = [v for v in intensities if v > self.silence_threshold_db]
                overall = float(np.mean(voiced)) if voiced else (float(np.mean(intensities)) if intensities else INT_FALLBACK_DB)

                if self._running:
                    self.result_ready.emit(overall, times, intensities)

        except Exception as e:
            print(f"[IntensityAnalyzer] Exception: {e}")

# ---------------------------------------------
# КАНВАС ДЛЯ ОТОБРАЖЕНИЯ ИНТЕНСИВНОСТИ
# ---------------------------------------------
class MatplotlibCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(9, 5))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self._setup()

    def _setup(self):
        self.ax.set_title("Интенсивность голоса, dB")
        self.ax.set_xlabel("Время (с)")
        self.ax.set_ylabel("dB")
        self.ax.grid(True)

    def plot(self, times: list[float], intensities_db: list[float]):
        self.ax.clear()
        self._setup()
        y = intensities_db or []
        self.ax.plot(times, y, label="Интенсивность (EMA)")
        # Динамический диапазон по данным
        if y:
            ymin = min(y); ymax = max(y)
            if np.isfinite([ymin, ymax]).all():
                pad = max(3.0, (ymax - ymin) * 0.1)
                if ymin == ymax:
                    ymin -= 5.0; ymax += 5.0
                self.ax.set_ylim(ymin - pad, ymax + pad)
        self.ax.legend()
        self.draw()

    def clear_plot(self):
        self.ax.clear()
        self._setup()
        self.draw()

# ---------------------------------------------
# ОКНО РЕДАКТОРА ЗАДАНИЯ (ИНТЕНСИВНОСТЬ, dB) ДЛЯ GROMIK
# ---------------------------------------------
class TaskEditor(QWidget):
    """
    Живёт внутри QStackedWidget.
    on_save() — вызывается после сохранения.
    on_close() — вызывается при закрытии без сохранения.
    """
    def __init__(self, task: Optional[dict] = None, on_save=None, on_close=None, profile_name: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Редактор задания (Интенсивность, dB) — Gromik")
        self.setMinimumWidth(1000)

        self.profile_name = profile_name
        self.task = task or {}
        self.on_save = on_save
        self.on_close = on_close
        self._saved = False

        self.canvas = MatplotlibCanvas()
        self.analyzer: Optional[IntensityAnalyzer] = None

        # --- Левая форма ---
        root = QHBoxLayout(self)
        form = QVBoxLayout()

        def add_labeled_input(label: str, key: str, default: str = ""):
            row = QHBoxLayout()
            lbl = QLabel(label)
            line = QLineEdit(str(self.task.get(key, default)))
            btn = QPushButton("Узнать (5 сек)")
            row.addWidget(lbl)
            row.addWidget(line)
            row.addWidget(btn)
            form.addLayout(row)
            return line, btn

        self.name_edit = QLineEdit(str(self.task.get("name", "")))
        form.addWidget(QLabel("Название задания:"))
        form.addWidget(self.name_edit)

        # dB уровни вместо Hz
        self.quiet_line, btn_quiet = add_labeled_input("Тихий (dB):", "quiet", "55")
        self.norm_line,  btn_norm  = add_labeled_input("Нормальный (dB):", "norm", "70")
        self.loud_line,  btn_loud  = add_labeled_input("Громкий (dB):", "loud", "80")

        # Валидаторы: допустим 20..120 dB
        for le in (self.quiet_line, self.norm_line, self.loud_line):
            le.setValidator(QDoubleValidator(20.0, 120.0, 1, self))

        # Генерация уровней
        self.chk_quiet = QCheckBox("Генерировать 'тихий' уровень")
        self.chk_quiet.setChecked(self.task.get("gen_quiet", False))
        form.addWidget(self.chk_quiet)

        self.chk_norm = QCheckBox("Генерировать 'нормальный' уровень")
        self.chk_norm.setChecked(self.task.get("gen_norm", True))
        form.addWidget(self.chk_norm)

        self.chk_loud = QCheckBox("Генерировать 'громкий' уровень")
        self.chk_loud.setChecked(self.task.get("gen_loud", True))
        form.addWidget(self.chk_loud)

        # Остальные параметры — совместимы с твоей игрой
        self.frequency_line = QLineEdit(str(self.task.get("frequency", "6")))
        self.duration_line = QLineEdit(str(self.task.get("duration", "60")))
        self.artifacts_count_line = QLineEdit(str(self.task.get("artifacts_count", "5")))
        self.artifact_interval_line = QLineEdit(str(self.task.get("artifact_interval", "0.2")))
        self.text_line = QLineEdit(str(self.task.get("text", "ДА")))
        self.smooth_chk = QCheckBox("Сглаженное управление (слитно)")
        self.smooth_chk.setChecked(self.task.get("smooth", True))

        self.frequency_line.setValidator(QDoubleValidator(0.1, 60.0, 2, self))
        self.duration_line.setValidator(QIntValidator(5, 3600, self))
        self.artifacts_count_line.setValidator(QIntValidator(1, 1000, self))
        self.artifact_interval_line.setValidator(QDoubleValidator(0.01, 10.0, 2, self))

        form.addWidget(QLabel("Частота появления заданий (с)"));      form.addWidget(self.frequency_line)
        form.addWidget(QLabel("Длительность игры (с)"));              form.addWidget(self.duration_line)
        form.addWidget(QLabel("Кол-во артефактов (в волне)"));        form.addWidget(self.artifacts_count_line)
        form.addWidget(QLabel("Интервал между артефактами (с)"));     form.addWidget(self.artifact_interval_line)
        form.addWidget(QLabel("Текст задания (слог)"));               form.addWidget(self.text_line)
        form.addWidget(self.smooth_chk)

        # Кнопки
        btns = QHBoxLayout()
        btn_save = QPushButton("Сохранить")
        btn_back = QPushButton("Назад")
        btns.addWidget(btn_save); btns.addWidget(btn_back)
        form.addLayout(btns)

        # Справа — график
        root.addLayout(form, 1)
        root.addWidget(self.canvas, 2)

        # Подключения
        for button, line_edit in ((btn_quiet, self.quiet_line), (btn_norm, self.norm_line), (btn_loud, self.loud_line)):
            button.clicked.connect(lambda _, le=line_edit: self.measure_intensity(le))
        btn_save.clicked.connect(self.save_task)
        btn_back.clicked.connect(self.close)

    # --------- Замер интенсивности (dB) ---------
    def measure_intensity(self, target_line: QLineEdit):
        # Остановим прежний поток, если был
        if self.analyzer is not None and self.analyzer.isRunning():
            self.analyzer.stop()
            self.analyzer.wait()

        self.canvas.clear_plot()

        audio = get_audio(self.profile_name) if self.profile_name else {"mic_device_index": None, "silence_threshold_db": 45.0}
        device_index = audio.get("mic_device_index", None)
        silence_thr = float(audio.get("silence_threshold_db", 45.0))

        self.analyzer = IntensityAnalyzer(device_index=device_index, silence_threshold_db=silence_thr)
        self.analyzer.setParent(self)

        self.analyzer.live_update.connect(
            self.plot_live,
            QtCore.Qt.ConnectionType.QueuedConnection
        )
        self.analyzer.result_ready.connect(
            lambda avg_db, times, intensities: self.handle_result(avg_db, times, intensities, target_line),
            QtCore.Qt.ConnectionType.QueuedConnection
        )
        self.analyzer.finished.connect(lambda: setattr(self, "analyzer", None))

        self.analyzer.start()

    def plot_live(self, times: list[float], intensities_db: list[float]):
        if not self.isVisible() or not hasattr(self, "canvas") or self.canvas is None:
            return
        try:
            self.canvas.plot(times, intensities_db)
        except RuntimeError:
            pass

    def handle_result(self, avg_db: float, times: list[float], intensities_db: list[float], target_line: QLineEdit):
        if not self.isVisible() or not hasattr(self, "canvas") or self.canvas is None:
            return
        # округлим до 1 dB — точнее и понятнее
        try:
            target_line.setText(str(round(float(avg_db), 1)))
        except Exception:
            target_line.setText(str(avg_db))
        try:
            self.canvas.plot(times, intensities_db)
        except RuntimeError:
            pass

    # --------- Закрытие окна / очистка потока ---------
    def closeEvent(self, event):
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

        if not self._saved and callable(self.on_close):
            cb = self.on_close
            QTimer.singleShot(0, lambda: cb())

        super().closeEvent(event)

    # --------- Сохранение ---------
    def save_task(self):
        def float_or(le: QLineEdit, default: float) -> float:
            txt = le.text().strip()
            try:
                return float(txt) if txt else default
            except ValueError:
                return default

        def int_or(le: QLineEdit, default: int) -> int:
            txt = le.text().strip()
            try:
                return int(txt) if txt else default
            except ValueError:
                return default

        name = self.name_edit.text().strip()

        new_task = {
            "name": name,
            # ВНИМАНИЕ: ключи такие же, как в pitch-задачах, но значения теперь в dB.
            "quiet": float_or(self.quiet_line, 55.0),
            "norm":  float_or(self.norm_line, 70.0),
            "loud":  float_or(self.loud_line, 80.0),

            "gen_quiet": self.chk_quiet.isChecked(),
            "gen_norm":  self.chk_norm.isChecked(),
            "gen_loud":  self.chk_loud.isChecked(),

            "frequency":         float_or(self.frequency_line, 6.0),
            "duration":          int_or(self.duration_line, 60),
            "artifacts_count":   int_or(self.artifacts_count_line, 5),
            "artifact_interval": float_or(self.artifact_interval_line, 0.2),
            "text":              self.text_line.text().strip(),
            "smooth":            self.smooth_chk.isChecked(),

            # Необязательно, но может пригодиться игре:
            "metric": "intensity_db"
        }

        if not self.profile_name:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Профиль не задан", "Редактор открыт без профиля")
        else:
            upsert_task(self.profile_name, new_task)
            self._saved = True
            if callable(self.on_save):
                try:
                    QTimer.singleShot(0, self.on_save)
                except Exception:
                    pass
            self.close()
