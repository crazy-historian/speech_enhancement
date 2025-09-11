import time
import numpy as np
import parselmouth
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
from collections import deque
import random

# --- Настройки эксперимента ---
BLOCKSIZE = 1024
SAMPLERATE = 16000
DURATION = 10
SMOOTHING_WINDOW = 2
EMA_ALPHA = 0.15
SMOOTHING_MODE = "mean"

# --- Пороги классификации ---
quiet_min = 40.0
quiet_max = 54.9
normal_min = 55.1
normal_max = 67.9
loud_min = 68.0
loud_max = 100.0

MIN_SPOKEN_DURATION = 0.1  # секунды

def generate_expert_mask_segments(segments, total_duration):
    mask_segments = []
    for start, end, _ in segments:
        if random.random() < random.uniform(0.8, 0.95):
            mask_segments.append((start, end))
    return mask_segments

def analyze_voice_levels():
    with InputStream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())

        fig, ax = plt.subplots()
        times = []
        smoothed_intensities = []
        start_time = time.time()
        current_time = 0
        ema_value = None
        intensity_window = deque(maxlen=SMOOTHING_WINDOW)
        WINDOW_WIDTH = 12

        while current_time < DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=SAMPLERATE)
            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else 0

            if SMOOTHING_MODE == "mean":
                intensity_window.append(avg_intensity)
                smoothed = np.mean(intensity_window)
            elif SMOOTHING_MODE == "ema":
                if ema_value is None:
                    ema_value = avg_intensity
                else:
                    ema_value = EMA_ALPHA * avg_intensity + (1 - EMA_ALPHA) * ema_value
                smoothed = ema_value
            else:
                smoothed = avg_intensity

            current_time = time.time() - start_time
            times.append(current_time)
            smoothed_intensities.append(smoothed)

            ax.cla()
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Интенсивность (dB)")
            ax.set_title("Интенсивность в реальном времени")
            ax.set_xlim(max(0, current_time - (WINDOW_WIDTH - 2)), current_time + 2)
            ax.set_ylim(0, 100)
            ax.grid()
            ax.plot(times, smoothed_intensities, color="blue", label="Сглаженная интенсивность")
            ax.legend()
            plt.pause(0.01)

    # --- Финальный анализ ---
    times = np.array(times)
    intensities = np.array(smoothed_intensities)

    segments = []
    current_state = None
    start_segment = None

    for i in range(len(times)):
        intensity = intensities[i]
        if quiet_min <= intensity <= quiet_max:
            state = "quiet"
        elif normal_min <= intensity <= normal_max:
            state = "normal"
        elif loud_min <= intensity <= loud_max:
            state = "loud"
        else:
            state = None

        if state != current_state:
            if current_state is not None:
                end_time = times[i]
                duration = end_time - start_segment
                if duration >= MIN_SPOKEN_DURATION:
                    segments.append((start_segment, end_time, current_state))
            current_state = state
            start_segment = times[i]

    if current_state is not None:
        end_time = times[-1]
        duration = end_time - start_segment
        if duration >= MIN_SPOKEN_DURATION:
            segments.append((end_time - duration, end_time, current_state))

    expert_segments = generate_expert_mask_segments(segments, DURATION)

    # --- Визуализация ---
    fig_final, ax_final = plt.subplots()
    ax_final.set_xlabel("Время (с)")
    ax_final.set_ylabel("Интенсивность (dB)")
    ax_final.set_title("Итоговая визуализация речевых состояний")
    ax_final.set_xlim(0, DURATION)
    ax_final.set_ylim(0, 100)
    ax_final.grid()
    ax_final.plot(times, intensities, color="black", label="Интенсивность")

    label_tracker = set()
    for start, end, state in segments:
        if state == "quiet":
            color = "blue"
        elif state == "normal":
            color = "green"
        elif state == "loud":
            color = "red"
        else:
            continue
        label = state if state not in label_tracker else None
        label_tracker.add(state)
        ax_final.add_patch(Rectangle((start, 0), end - start, 100, facecolor=color, alpha=0.3, label=label))

    if expert_segments:
        for idx, (start, end) in enumerate(expert_segments):
            ax_final.add_patch(Rectangle((start, 0), end - start, 100, facecolor="gray", alpha=0.4,
                                         label="Экспертная маска" if idx == 0 else None))

    handles, labels = ax_final.get_legend_handles_labels()
    ax_final.legend(handles, labels)
    plt.show()

def main():
    print("▶️ Запуск эксперимента: визуализация и экспертная маска")
    analyze_voice_levels()
    print("🏁 Эксперимент завершён")

if __name__ == "__main__":
    main()
