# pitch_guiconfig.py
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

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton,
    QLineEdit, QHBoxLayout, QCheckBox, QListWidget, QMessageBox, QGridLayout, QDialog, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QEventLoop

# ---------------------------------------------
# ЕДИНЫЙ ФАЙЛ (pitch_config.json) ДЛЯ ВСЕГО
# ---------------------------------------------
PITCH_FILE = Path("profiles/pitch_config.json")

def load_pitch_config():
    """
    Загружает конфигурацию (mic_device_index) и tasks
    из pitch_config.json.
    Возвращает dict вида:
    {
      "mic_device_index": int или None,
      "tasks": [ { ... }, { ... } ]
    }
    """
    if PITCH_FILE.exists():
        return json.loads(PITCH_FILE.read_text(encoding="utf-8"))
    else:
        # если файл ещё не создан — вернуть пустую структуру
        return {
            "mic_device_index": None,
            "tasks": []
        }

def save_pitch_config(data: dict):
    """
    Сохраняет конфигурацию в pitch_config.json
    """
    PITCH_FILE.parent.mkdir(parents=True, exist_ok=True)  # убедимся, что папка profiles создана
    PITCH_FILE.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------
# ПОТОК ДЛЯ АНАЛИЗА PITCH В РЕАЛЬНОМ ВРЕМЕНИ
# ---------------------------------------------
BLOCKSIZE = 1024
RECORD_DURATION = 5  # длина записи (сек) при замерах
PITCH_FLOOR = 100
PITCH_CEILING = 600
VOICING_THRESHOLD = 0.6
SILENCE_THRESHOLD_DB = 45.0
BLOCKS_TO_SILENT = 2

class PitchAnalyzer(QThread):
    result_ready = pyqtSignal(float, list, list)
    live_update = pyqtSignal(list, list)

    def __init__(self, device_index=None):
        super().__init__()
        self._running = True
        self.device_index = device_index

    def stop(self):
        self._running = False

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
                current_time = 0
                silent_counter = 0
                last_valid_pitch = None

                while (time.time() - start_time < RECORD_DURATION) and self._running:
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
        self.fig = Figure(figsize=(5, 4))
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
    def __init__(self, task=None, on_save=None):
        super().__init__()
        self.setWindowTitle("Редактор задания (Pitch)")
        self.setMinimumWidth(1000)
        self._closed = False  # флаг закрытия окна

        self.task = task or {}
        self.on_save = on_save
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
        self.norm_line, btn_norm = add_labeled_input("Нормальный (Hz):", "norm", "180")
        self.loud_line, btn_loud = add_labeled_input("Громкий (Hz):", "loud", "240")

        self.pitch_buttons = [
            (btn_quiet, self.quiet_line),
            (btn_norm, self.norm_line),
            (btn_loud, self.loud_line),
        ]

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
        form_layout.addWidget(QLabel("Частота появления заданий (с)"))
        form_layout.addWidget(self.frequency_line)

        self.duration_line = QLineEdit(str(self.task.get("duration", "60")))
        form_layout.addWidget(QLabel("Длительность игры (с)"))
        form_layout.addWidget(self.duration_line)

        self.artifacts_count_line = QLineEdit(str(self.task.get("artifacts_count", "5")))
        form_layout.addWidget(QLabel("Кол-во артефактов (в волне)"))
        form_layout.addWidget(self.artifacts_count_line)

        self.artifact_interval_line = QLineEdit(str(self.task.get("artifact_interval", "0.2")))
        form_layout.addWidget(QLabel("Интервал между артефактами (с)"))
        form_layout.addWidget(self.artifact_interval_line)

        self.text_line = QLineEdit(str(self.task.get("text", "ДА")))
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

        layout.addLayout(form_layout, 2)
        layout.addWidget(self.canvas, 3)
        self.setLayout(layout)

        for button, line_edit in self.pitch_buttons:
            button.clicked.connect(lambda _, le=line_edit: self.measure_pitch(le))

        btn_save.clicked.connect(self.save_task)
        btn_back.clicked.connect(self.close)

    def measure_pitch(self, target_line):
        # Если уже работает предыдущий анализатор, останавливаем его
        if self.analyzer is not None and self.analyzer.isRunning():
            self.analyzer.stop()
            self.analyzer.wait()

        if self.canvas:
            self.canvas.clear_plot()

        conf = load_pitch_config()
        device_index = conf.get("mic_device_index", None)

        self.analyzer = PitchAnalyzer(device_index=device_index)
        self.analyzer.setParent(self)  # для корректного управления памятью
        self.analyzer.live_update.connect(self.plot_live)
        self.analyzer.result_ready.connect(
            lambda avg_pitch, times, pitches: self.handle_result(avg_pitch, times, pitches, target_line)
        )
        self.analyzer.finished.connect(lambda: setattr(self, "analyzer", None))
        self.analyzer.start()

    def plot_live(self, times, pitches):
        if not self.isVisible() or not hasattr(self, "canvas") or self.canvas is None:
            return
        try:
            self.canvas.plot(times, pitches)
        except RuntimeError:
            # Если canvas уже удалён – ничего не делаем
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
        if self.analyzer is not None:
            try:
                self.analyzer.live_update.disconnect()
                self.analyzer.result_ready.disconnect()
            except Exception:
                pass
            if self.analyzer.isRunning():
                self.analyzer.stop()
                self.analyzer.wait()
            self.analyzer = None
        # Не вызываем self.canvas.deleteLater() и не обнуляем self.canvas – пусть Qt управляет этим автоматически.
        super().closeEvent(event)

    def save_task(self):
        name = self.name_edit.text().strip()
        quiet_val = int(self.quiet_line.text().strip())
        norm_val = int(self.norm_line.text().strip())
        loud_val = int(self.loud_line.text().strip())
        data = load_pitch_config()

        new_task = {
            "name": name,
            "quiet": quiet_val,
            "norm": norm_val,
            "loud": loud_val,
            "gen_quiet": self.chk_quiet.isChecked(),
            "gen_norm": self.chk_norm.isChecked(),
            "gen_loud": self.chk_loud.isChecked(),
            "frequency": float(self.frequency_line.text()),
            "duration": int(self.duration_line.text()),
            "artifacts_count": int(self.artifacts_count_line.text()),
            "artifact_interval": float(self.artifact_interval_line.text()),
            "text": self.text_line.text().strip(),
            "smooth": self.smooth_chk.isChecked()
        }

        tasks = data.get("tasks", [])
        found = False
        for i, t in enumerate(tasks):
            if t.get("name") == name:
                tasks[i] = new_task
                found = True
                break
        if not found:
            tasks.append(new_task)
        data["tasks"] = tasks

        save_pitch_config(data)
        if self.on_save:
            self.on_save()
        self.close()



# ---------------------------------------------
# ГЛАВНОЕ ОКНО (список заданий, выбор микрофона)
# ---------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Настройки игры (Pitch)")
        layout = QVBoxLayout()

        # 1) Выбор микрофона
        self.device_selector = QComboBox()
        self.devices = AudioDevices()
        self.device_map = {}

        # Получим словарь {dev_id: dev_info}
        input_devices_info = self.devices.get_hostapi_with_devices(kind='input')
        input_devices = []
        for api_name, devs in input_devices_info.items():
            for dev_id, dev_info in devs.items():
                input_devices.append({
                    "name": dev_info["name"],
                    "index": int(dev_id)
                })
        # Заполним comboBox
        for dev in input_devices:
            self.device_selector.addItem(dev["name"])
            self.device_map[dev["name"]] = dev["index"]

        layout.addWidget(QLabel("Выберите микрофон:"))
        layout.addWidget(self.device_selector)

        # загрузим сохранённый device_index и проставим его
        config = load_pitch_config()
        saved_index = config.get("mic_device_index")
        if saved_index is not None:
            for name, idx in self.device_map.items():
                if idx == saved_index:
                    self.device_selector.setCurrentText(name)
                    break

        # 2) Список заданий
        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self.edit_task)
        layout.addWidget(self.task_list)

        # Кнопка "Добавить задание"
        btn_add = QPushButton("Добавить задание")
        btn_add.clicked.connect(self.add_task)
        layout.addWidget(btn_add)

        # Кнопка "Старт игры"
        btn_start = QPushButton("Старт игры")
        btn_start.clicked.connect(self.start_game)
        layout.addWidget(btn_start)

        self.setLayout(layout)
        self.load_tasks()

    def load_tasks(self):
        self.task_list.clear()
        data = load_pitch_config()
        tasks = data.get("tasks", [])
        for t in tasks:
            self.task_list.addItem(t["name"])

    def get_selected_task(self):
        selected_item = self.task_list.currentItem()
        if not selected_item:
            return None

        name = selected_item.text()
        data = load_pitch_config()
        tasks = data.get("tasks", [])
        for t in tasks:
            if t["name"] == name:
                return t
        return None

    def edit_task(self):
        task = self.get_selected_task()
        if task:
            editor = TaskEditor(task, on_save=self.load_tasks)
            editor.show()

    def add_task(self):
        editor = TaskEditor(on_save=self.load_tasks)
        editor.show()

    def start_game(self):
        """
        Здесь логика запуска игры на Pitch.
        По аналогии с intensity-лаунчером ты можешь закрывать окно
        и вызывать run_pitch_game_with_task(task).
        """
        task = self.get_selected_task()
        if not task:
            QMessageBox.warning(self, "Нет задания", "Выберите задание для запуска игры")
            return

        # Сохраняем выбранный микрофон
        selected_name = self.device_selector.currentText()
        selected_index = self.device_map.get(selected_name)
        data = load_pitch_config()
        data["mic_device_index"] = selected_index
        save_pitch_config(data)

        print("Запуск игры (Pitch) с заданием:", task)
        # Закрой окно GUI
        self.close()
        # Здесь вызовите свою функцию, аналогичную run_game_with_task:
        # run_pitch_game_with_task(task)
        # (Это уже нужно дописать в твоём "launcher_pitch.py" или как-то иначе.)

# ---------------------------------------------
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ВЫБОРА ЗАДАНИЯ
# (Аналог select_task_gui, но для pitch)
# ---------------------------------------------
def select_task_gui():
    from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QListWidget
    class TaskSelector(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Выбор задания (Pitch)")
            self.setMinimumWidth(400)
            layout = QVBoxLayout()

            self.task_list = QListWidget()
            layout.addWidget(self.task_list)

            self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            layout.addWidget(self.button_box)
            self.setLayout(layout)

            data = load_pitch_config()
            tasks = data.get("tasks", [])
            for t in tasks:
                self.task_list.addItem(t["name"])

            self.button_box.accepted.connect(self.accept)
            self.button_box.rejected.connect(self.reject)

        def get_selected_task(self):
            selected_item = self.task_list.currentItem()
            if not selected_item:
                return None
            name = selected_item.text()

            data = load_pitch_config()
            for t in data.get("tasks", []):
                if t["name"] == name:
                    return t
            return None

    # Если QApplication ещё не создан — создадим
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    dialog = TaskSelector()
    result = dialog.exec()
    if result == QDialog.DialogCode.Accepted:
        return dialog.get_selected_task()
    return None


# ---------------------------------------------
# ЗАПУСК АПП (если запускать напрямую)
# ---------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
