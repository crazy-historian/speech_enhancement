import time
import numpy as np
import random
import parselmouth
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
from collections import deque

# --- Константы ---
BLOCKSIZE = 1024
SAMPLERATE = 16000
GAME_DURATION = 20
TASK_INTERVAL = 6
TASK_DURATION = 2
INTENSITY_RANGE = (55, 56)
TOLERANCE_DB = 10
SMOOTHING_WINDOW = 5  # для "mean"
EMA_ALPHA = 0.15   # для "ema"

# Переключатель режима сглаживания
SMOOTHING_MODE = "mean"  # или "ema"

# --- Генерация заданий ---
def generate_tasks():
    tasks = []
    t = TASK_INTERVAL
    while t + TASK_DURATION <= GAME_DURATION:
        center_db = random.uniform(*INTENSITY_RANGE)
        low = center_db - TOLERANCE_DB
        high = center_db + TOLERANCE_DB
        tasks.append({'start': t, 'end': t + TASK_DURATION, 'low': low, 'high': high})
        t += TASK_INTERVAL
    return tasks

# --- Основная функция ---
def analyze_and_plot_voice_intensity():
    tasks = generate_tasks()

    with InputStream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())

        fig, ax = plt.subplots()
        times = []
        raw_intensities = []
        smoothed_intensities = []
        start_time = time.time()
        current_time = 0
        WINDOW_WIDTH = 12

        # Для сглаживания
        ema_value = None
        intensity_window = deque(maxlen=SMOOTHING_WINDOW)

        while current_time < GAME_DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=SAMPLERATE)

            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else 0
            raw_intensities.append(avg_intensity)
            times.append(current_time)

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

            smoothed_intensities.append(smoothed)

            # --- Отрисовка ---
            ax.cla()
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Интенсивность (dB)")
            ax.set_title(f"Интенсивность (режим: {SMOOTHING_MODE})")
            ax.set_xlim(max(0, current_time - (WINDOW_WIDTH - 2)), current_time + 2)
            ax.set_ylim(0, 100)
            ax.grid()

            for task in tasks:
                ax.add_patch(Rectangle(
                    (task['start'], task['low']),
                    TASK_DURATION,
                    task['high'] - task['low'],
                    facecolor='lightgray',
                    edgecolor='gray',
                    alpha=0.4
                ))

            ax.plot(times, smoothed_intensities, color="red", label="Сглаженная интенсивность")
            ax.legend()
            plt.pause(0.01)

            current_time += BLOCKSIZE / SAMPLERATE

    # --- Анализ попаданий в задания ---
    times = np.array(times)
    intensities = np.array(smoothed_intensities)
    successful, total = 0, len(tasks)
    task_results = []

    for task in tasks:
        indices = np.where((times >= task['start']) & (times < task['end']))[0]
        in_range = np.all((intensities[indices] >= task['low']) & (intensities[indices] <= task['high']))
        task_results.append((task, in_range))
        if in_range:
            successful += 1

    print(f"\n✅ Успешных выполнений: {successful} из {total}")

    # --- Финальный график ---
    fig_final, ax_final = plt.subplots()
    ax_final.set_xlabel("Время (с)")
    ax_final.set_ylabel("Интенсивность (dB)")
    ax_final.set_title("Итоговая визуализация интенсивности")
    ax_final.set_xlim(0, GAME_DURATION)
    ax_final.set_ylim(0, 100)
    ax_final.grid()

    ax_final.plot(times, intensities, color="darkred", label="Сглаженная интенсивность")

    for task, success in task_results:
        color = 'green' if success else 'red'
        ax_final.add_patch(Rectangle(
            (task['start'], task['low']),
            TASK_DURATION,
            task['high'] - task['low'],
            facecolor=color,
            alpha=0.2
        ))

    ax_final.legend()
    plt.show()

def main():
    print("Запуск анализа интенсивности с режимом сглаживания...")
    analyze_and_plot_voice_intensity()
    print("Анализ завершён.")

if __name__ == "__main__":
    main()
