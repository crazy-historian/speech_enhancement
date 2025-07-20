import time
import numpy as np
import parselmouth
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
from collections import deque

# --- Настройки ---
BLOCKSIZE = 1024
SAMPLERATE = 16000
DURATION = 10
SMOOTHING_WINDOW = 2
EMA_ALPHA = 0.15
SMOOTHING_MODE = "mean"  # "mean" или "ema"
MIN_PITCH_SEGMENT_DURATION = 0.1  # сек
SILENCE_THRESHOLD_DB = 45.0

# --- Диапазоны частот ---
low_min = 80
low_max = 140
normal_min = 140.1
normal_max = 210
high_min = 210.1
high_max = 600

def analyze_pitch_levels():
    with InputStream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())

        fig, ax = plt.subplots()
        times = []
        smoothed_pitches = []
        start_time = time.time()
        current_time = 0
        ema_value = None
        pitch_window = deque(maxlen=SMOOTHING_WINDOW)
        WINDOW_WIDTH = 12

        while current_time < DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=SAMPLERATE)

            # --- Интенсивность ---
            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

            # --- Если тишина — pitch = 0 ---
            if avg_intensity < SILENCE_THRESHOLD_DB:
                avg_pitch = 0
            else:
                pitch_obj = sound.to_pitch_ac(
                    time_step=0.01,
                    pitch_floor=75,
                    pitch_ceiling=600,
                    voicing_threshold=0.6
                )
                pitch_values = pitch_obj.selected_array['frequency']
                pitch_values[(pitch_values == 0) | (pitch_values > 600)] = np.nan
                avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else 0

            # --- Сглаживание ---
            if SMOOTHING_MODE == "mean":
                pitch_window.append(avg_pitch)
                smoothed = np.mean(pitch_window)
            elif SMOOTHING_MODE == "ema":
                if ema_value is None:
                    ema_value = avg_pitch
                else:
                    ema_value = EMA_ALPHA * avg_pitch + (1 - EMA_ALPHA) * ema_value
                smoothed = ema_value
            else:
                smoothed = avg_pitch

            current_time = time.time() - start_time
            times.append(current_time)
            smoothed_pitches.append(smoothed)

            # --- Живой график ---
            ax.cla()
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Частота (Гц)")
            ax.set_title("Pitch в реальном времени")
            ax.set_xlim(max(0, current_time - (WINDOW_WIDTH - 2)), current_time + 2)
            ax.set_ylim(75, 600)
            ax.grid()
            ax.plot(times, smoothed_pitches, color="purple", label="Сглаженный pitch")
            ax.legend()
            plt.pause(0.01)

    # --- Финальный анализ ---
    times = np.array(times)
    pitches = np.array(smoothed_pitches)

    segments = []
    current_state = None
    start_segment = None

    for i in range(len(times)):
        pitch = pitches[i]
        if low_min <= pitch <= low_max:
            state = "low"
        elif normal_min <= pitch <= normal_max:
            state = "normal"
        elif high_min <= pitch <= high_max:
            state = "high"
        else:
            state = None

        if state != current_state:
            if current_state is not None:
                end_time = times[i]
                duration = end_time - start_segment
                if duration >= MIN_PITCH_SEGMENT_DURATION:
                    segments.append((start_segment, end_time, current_state))
            current_state = state
            start_segment = times[i]

    if current_state is not None:
        end_time = times[-1]
        duration = end_time - start_segment
        if duration >= MIN_PITCH_SEGMENT_DURATION:
            segments.append((start_segment, end_time, current_state))

    # --- Подсчёт по категориям ---
    summary = {"low": 0.0, "normal": 0.0, "high": 0.0}
    for start, end, state in segments:
        if state:
            summary[state] += end - start

    # --- Финальный график ---
    fig_final, ax_final = plt.subplots()
    ax_final.set_xlabel("Время (с)")
    ax_final.set_ylabel("Частота (Гц)")
    ax_final.set_title("Итоговая визуализация pitch-состояний")
    ax_final.set_xlim(0, DURATION)
    ax_final.set_ylim(75, 600)
    ax_final.grid()
    ax_final.plot(times, pitches, color="black", label="Pitch")

    for start, end, state in segments:
        color = {"low": "blue", "normal": "green", "high": "red"}.get(state, "gray")
        ax_final.add_patch(Rectangle(
            (start, 75),
            end - start,
            600 - 75,
            facecolor=color,
            alpha=0.3,
            label=state if not any(p.get_label() == state for p in ax_final.patches) else None
        ))

    handles, labels = ax_final.get_legend_handles_labels()
    ax_final.legend(handles, labels)
    plt.show()

    # --- Сводка ---
    print(f"\n📊 Сводка по pitch (учитываются только участки ≥ {MIN_PITCH_SEGMENT_DURATION} с):")
    for state in ["low", "normal", "high"]:
        print(f"🔹 {state.capitalize():<7}: {summary[state]:.1f} секунд")

def main():
    print("🎙️ Запуск pitch-анализатора с фильтрацией тишины")
    analyze_pitch_levels()
    print("✅ Анализ завершён")

if __name__ == "__main__":
    main()
