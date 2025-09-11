
from __future__ import annotations
from typing import Optional

from PyQt6.QtCore import QTimer, QRegularExpression
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton, QLabel
from PyQt6.QtGui import QRegularExpressionValidator

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .analyzers import VoiceFlowAnalyzer
from .profiles import get_audio, upsert_task


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
        self.ax.plot(times, values)
        for pause_start, pause_duration in pauses:
            self.ax.axvspan(pause_start, pause_start + pause_duration, alpha=0.3)
            self.ax.text(
                pause_start + pause_duration / 2,
                1.05,
                f"{pause_duration:.2f}s",
                ha="center",
                fontsize=8,
            )
        self.draw()


# --------------------------- Окно настройки задания ---------------------------
class TaskEditor(QWidget):
    """
    Дружественная к вводу версия:
    - Валидаторы на regex допускают пустые строки и промежуточные состояния.
    - При сохранении запятая автоматически конвертируется в точку.
    """
    def __init__(self, task: Optional[dict] = None, profile_name: Optional[str] = None,
                 on_save=None, on_close=None):
        super().__init__()
        self.setWindowTitle("Редактор задания (slogotakt)")
        self.setMinimumWidth(900)
        self.profile_name = profile_name
        self.task = task or {}
        self.on_save = on_save
        self.on_close = on_close
        self._saved = False

        self.canvas = FlowPlotCanvas()
        self.analyzer: Optional[VoiceFlowAnalyzer] = None

        layout = QHBoxLayout(self)
        left = QVBoxLayout()

        # Поля
        self.name_input = QLineEdit(self.task.get("name", ""))
        self.duration_input = QLineEdit(str(self.task.get("duration", 60)))
        self.wave_interval_input = QLineEdit(str(self.task.get("wave_interval", 3)))
        self.wave_size_input = QLineEdit(str(self.task.get("wave_size", 3)))
        self.syllable_interval_input = QLineEdit(str(self.task.get("syllable_interval", 0.5)))
        self.text_input = QLineEdit(self.task.get("text", "МА"))

        # --- Валидаторы на регулярках (разрешают и '.' и ',',
        #     и пустые строки, и промежуточное состояние) ---
        int_re = QRegularExpression(r"^$|^\d{1,4}$")
        float_re = QRegularExpression(r"^$|^\d{0,4}([.,]\d{0,3})?$")

        self.duration_input.setValidator(QRegularExpressionValidator(float_re, self))
        self.wave_interval_input.setValidator(QRegularExpressionValidator(float_re, self))
        self.wave_size_input.setValidator(QRegularExpressionValidator(int_re, self))
        self.syllable_interval_input.setValidator(QRegularExpressionValidator(float_re, self))

        # Разметка
        left.addWidget(QLabel("Название задания:")); left.addWidget(self.name_input)
        left.addWidget(QLabel("Длительность игры (сек):")); left.addWidget(self.duration_input)
        left.addWidget(QLabel("Интервал между волнами (сек):")); left.addWidget(self.wave_interval_input)
        left.addWidget(QLabel("Кол-во слогов в волне:")); left.addWidget(self.wave_size_input)
        left.addWidget(QLabel("Интервал между слогами (сек):")); left.addWidget(self.syllable_interval_input)

        test_btn = QPushButton("Узнать")
        test_btn.clicked.connect(self.start_measurement)
        left.addWidget(test_btn)

        left.addWidget(QLabel("Текст задания (слоги):")); left.addWidget(self.text_input)

        btns = QHBoxLayout()
        btn_save = QPushButton("Сохранить")
        btn_back = QPushButton("Назад")
        btn_save.clicked.connect(self.save_task)
        btn_back.clicked.connect(self.close)
        btns.addWidget(btn_save)
        btns.addWidget(btn_back)
        left.addLayout(btns)

        layout.addLayout(left, 1)
        layout.addWidget(self.canvas, 2)

    # ---- измерение ----
    def start_measurement(self):
        if self.analyzer is not None and self.analyzer.isRunning():
            try:
                self.analyzer.stop()
                self.analyzer.wait()
            except Exception:
                pass
        audio = get_audio(self.profile_name) if self.profile_name else {"mic_device_index": None}
        dev = audio.get("mic_device_index")
        self.analyzer = VoiceFlowAnalyzer(device_index=dev)
        self.analyzer.update_plot.connect(self.canvas.update_graph)
        self.analyzer.average_pause_ready.connect(self.set_average_syllable_interval)
        self.analyzer.start()

    def set_average_syllable_interval(self, avg: float):
        self.syllable_interval_input.setText(f"{round(float(avg), 2)}")

    # ---- утилиты конвертации ----
    def _float_or(self, le: QLineEdit, default: float) -> float:
        txt = le.text().strip().replace(",", ".")
        return float(txt) if txt else default

    def _int_or(self, le: QLineEdit, default: int) -> int:
        txt = le.text().strip()
        return int(txt) if txt else default

    # ---- сохранение ----
    def save_task(self):
        new_task = {
            "name": (self.name_input.text().strip() or "Без названия"),
            "duration": self._float_or(self.duration_input, 60.0),
            "wave_interval": self._float_or(self.wave_interval_input, 3.0),
            "wave_size": self._int_or(self.wave_size_input, 3),
            "syllable_interval": self._float_or(self.syllable_interval_input, 0.5),
            "text": (self.text_input.text().strip() or "МА"),
        }
        if not self.profile_name:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Профиль не задан", "Редактор открыт без профиля")
            return
        upsert_task(self.profile_name, new_task)
        self._saved = True
        if callable(self.on_save):
            QTimer.singleShot(0, self.on_save)
        self.close()

    def closeEvent(self, event):
        if self.analyzer is not None and self.analyzer.isRunning():
            try:
                self.analyzer.stop()
                self.analyzer.wait(1000)
            except Exception:
                pass
        if not getattr(self, "_saved", False) and callable(self.on_close):
            QTimer.singleShot(0, self.on_close)
        super().closeEvent(event)
