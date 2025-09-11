from PyQt6.QtCore import QThread, pyqtSignal
import time
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

# ----------------------------- Константы -----------------------------
BLOCKSIZE = 1024
RECORD_DURATION = 5
PITCH_FLOOR = 100
PITCH_CEILING = 600
VOICING_THRESHOLD = 0.6
SILENCE_THRESHOLD_DB = 45.0
BLOCKS_TO_SILENT = 1

# ----------------------------- Анализатор интенсивности -----------------------------
class IntensityAnalyzer(QThread):
    result_ready = pyqtSignal(float, list, list)
    live_update = pyqtSignal(list, list)

    def __init__(self, device_index=None):
        super().__init__()
        self._running = True
        self.device_index = device_index

    def stop(self):
        self._running = False

    def run(self):
        with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2, device=self.device_index) as stream:
            stream.set_methods(UnpackRawInFloat32())
            start_time = time.time()
            times, intensities = [], []
            current_time = 0.0

            while time.time() - start_time < RECORD_DURATION and self._running:
                raw_data = stream.read(BLOCKSIZE)
                if not raw_data:
                    continue
                signal = stream.chain_of_methods(raw_data)
                sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)
                intensity_values = sound.to_intensity().values.T.flatten()
                avg_intensity = float(np.mean(intensity_values)) if len(intensity_values) > 0 else -50.0

                times.append(current_time)
                intensities.append(avg_intensity)
                current_time += BLOCKSIZE / 16000
                self.live_update.emit(times.copy(), intensities.copy())

            if intensities:
                overall_avg = float(np.mean(intensities))
                self.result_ready.emit(overall_avg, times, intensities)

# ----------------------------- Анализатор слитности -----------------------------
class VoiceFlowAnalyzer(QThread):
    update_plot = pyqtSignal(list, list, list)
    average_pause_ready = pyqtSignal(float)

    def __init__(self, device_index=None):
        super().__init__()
        self._running = True
        self.device_index = device_index

    def stop(self):
        self._running = False

    def run(self):
        with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2, device=self.device_index) as stream:
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

                intensity_values = sound.to_intensity().values.T.flatten()
                avg_intensity = float(np.mean(intensity_values)) if len(intensity_values) > 0 else -50.0

                pitch_obj = sound.to_pitch_ac(time_step=0.01, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING, voicing_threshold=VOICING_THRESHOLD)
                pitch_values = pitch_obj.selected_array['frequency']
                pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
                avg_pitch = float(np.nanmean(pitch_values)) if np.any(~np.isnan(pitch_values)) else None

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
                avg_pause = float(sum(pause_durations) / len(pause_durations))
                self.average_pause_ready.emit(avg_pause)