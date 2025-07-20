import time
import numpy as np
import parselmouth
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
from collections import deque

# --- Настройки эксперимента ---
BLOCKSIZE = 1024
SAMPLERATE = 16000
DURATION = 20
SMOOTHING_WINDOW = 1
EMA_ALPHA = 0.05
SMOOTHING_MODE = "mean"  # "mean" или "ema"

# --- Пороги классификации ---
quiet_min = 40.0
quiet_max = 55.0
normal_min = 56.0
normal_max = 64.9
loud_min = 65.0
loud_max = 100.0

# --- Минимальная продолжительность озвученного сегмента ---
MIN_SPOKEN_DURATION = 0.2  # секунды

# --- Основной анализ ---
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

            # --- Сглаживание ---
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
                smoothed = avg_intensity  # fallback

            current_time = time.time() - start_time
            times.append(current_time)
            smoothed_intensities.append(smoothed)

            # --- Живой график ---
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
            segments.append((start_segment, end_time, current_state))

    # --- Подсчёт времени по каждому состоянию ---
    summary = {"quiet": 0.0, "normal": 0.0, "loud": 0.0}
    for start, end, state in segments:
        if state:
            summary[state] += end - start

    # --- Финальный график ---
    fig_final, ax_final = plt.subplots()
    ax_final.set_xlabel("Время (с)")
    ax_final.set_ylabel("Интенсивность (dB)")
    ax_final.set_title("Итоговая визуализация речевых состояний")
    ax_final.set_xlim(0, DURATION)
    ax_final.set_ylim(0, 100)
    ax_final.grid()
    ax_final.plot(times, intensities, color="black", label="Интенсивность")

    for start, end, state in segments:
        if state == "quiet":
            color = "blue"
        elif state == "normal":
            color = "green"
        elif state == "loud":
            color = "red"
        else:
            continue

        ax_final.add_patch(Rectangle(
            (start, 0),
            end - start,
            100,
            facecolor=color,
            alpha=0.3,
            label=state if not any(s.get_label() == state for s in ax_final.patches) else None
        ))

    handles, labels = ax_final.get_legend_handles_labels()
    ax_final.legend(handles, labels)
    plt.show()

    # --- Сводка ---
    print(f"\n🧾 Сводка по состояниям (учитываются только участки ≥ {MIN_SPOKEN_DURATION} с):")
    for state in ["quiet", "normal", "loud"]:
        print(f"🔹 {state.capitalize():<7}: {summary[state]:.1f} секунд")

def main():
    print("▶️ Запуск эксперимента 2: классификация речевых участков")
    analyze_voice_levels()
    print("🏁 Эксперимент завершён")

if __name__ == "__main__":
    main()