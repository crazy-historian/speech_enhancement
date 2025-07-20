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
GAME_DURATION = 60
TASK_INTERVAL = 6
TASK_DURATION = 2
PITCH_RANGE = (130, 140)
PITCH_TOLERANCE = 20
SMOOTHING_WINDOW = 1
EMA_ALPHA = 0.15
SMOOTHING_MODE = "mean"  # "mean" или "ema"
SILENCE_THRESHOLD_DB = 45.0
BLOCKS_TO_SILENT = 3

def generate_pitch_tasks():
    tasks = []
    t = TASK_INTERVAL
    while t + TASK_DURATION <= GAME_DURATION:
        center_pitch = random.uniform(*PITCH_RANGE)
        low = center_pitch - PITCH_TOLERANCE
        high = center_pitch + PITCH_TOLERANCE
        tasks.append({'start': t, 'end': t + TASK_DURATION, 'low': low, 'high': high})
        t += TASK_INTERVAL
    return tasks

def analyze_and_plot_pitch():
    tasks = generate_pitch_tasks()

    with InputStream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())

        fig, ax = plt.subplots()
        times, raw_pitches, smoothed_pitches = [], [], []
        start_time = time.time()
        current_time = 0
        WINDOW_WIDTH = 12

        ema_value = None
        pitch_window = deque(maxlen=SMOOTHING_WINDOW)
        silent_counter = 0
        last_valid_pitch = 0

        while current_time < GAME_DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=SAMPLERATE)

            # --- Интенсивность ---
            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

            # --- Частота ---
            pitch_obj = sound.to_pitch_ac(
                time_step=0.01,
                pitch_floor=100,
                pitch_ceiling=600,
                voicing_threshold=0.6
            )
            pitch_values = pitch_obj.selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > 600)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else 0

            # --- Учет тишины и сброса частоты ---
            if avg_intensity >= SILENCE_THRESHOLD_DB:
                silent_counter = 0
                last_valid_pitch = avg_pitch
            else:
                silent_counter += 1
                if silent_counter >= BLOCKS_TO_SILENT:
                    last_valid_pitch = 0  # обнуляем, если тишина продолжается

            raw_pitches.append(last_valid_pitch)
            times.append(current_time)

            # --- Сглаживание ---
            if SMOOTHING_MODE == "mean":
                pitch_window.append(last_valid_pitch)
                smoothed = np.mean(pitch_window)
            elif SMOOTHING_MODE == "ema":
                if ema_value is None:
                    ema_value = last_valid_pitch
                else:
                    ema_value = EMA_ALPHA * last_valid_pitch + (1 - EMA_ALPHA) * ema_value
                smoothed = ema_value
            else:
                smoothed = last_valid_pitch

            smoothed_pitches.append(smoothed)

            # --- Визуализация в реальном времени ---
            ax.cla()
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Частота (Гц)")
            ax.set_title(f"Основной тон (режим: {SMOOTHING_MODE})")
            ax.set_xlim(max(0, current_time - (WINDOW_WIDTH - 2)), current_time + 2)
            ax.set_ylim(80, 300)
            ax.grid()

            for task in tasks:
                ax.add_patch(Rectangle(
                    (task['start'], task['low']),
                    TASK_DURATION,
                    task['high'] - task['low'],
                    facecolor='lightblue',
                    edgecolor='gray',
                    alpha=0.4
                ))

            ax.plot(times, smoothed_pitches, color="blue", label="Сглаженная частота")
            ax.legend()
            plt.pause(0.01)

            current_time += BLOCKSIZE / SAMPLERATE

    # --- Анализ попаданий в задания ---
    times = np.array(times)
    pitches = np.array(smoothed_pitches)
    successful, total = 0, len(tasks)
    task_results = []

    print("\n🧠 Результаты по заданиям:\n")

    for idx, task in enumerate(tasks, 1):
        start, end, low, high = task['start'], task['end'], task['low'], task['high']
        indices = np.where((times >= start) & (times < end))[0]

        total_time = len(indices) * (BLOCKSIZE / SAMPLERATE)
        in_range_mask = (pitches[indices] >= low) & (pitches[indices] <= high)
        in_range_time = np.sum(in_range_mask) * (BLOCKSIZE / SAMPLERATE)
        in_range_pct = (in_range_time / total_time) * 100 if total_time > 0 else 0

        # Доп. маски
        low_mask = pitches[indices] < low
        normal_mask = (pitches[indices] >= low) & (pitches[indices] <= high)
        high_mask = pitches[indices] > high

        low_time = np.sum(low_mask) * (BLOCKSIZE / SAMPLERATE)
        normal_time = np.sum(normal_mask) * (BLOCKSIZE / SAMPLERATE)
        high_time = np.sum(high_mask) * (BLOCKSIZE / SAMPLERATE)

        low_pct = (low_time / total_time) * 100
        normal_pct = (normal_time / total_time) * 100
        high_pct = (high_time / total_time) * 100

        dominant_zone = max(
            [("Низко", low_time), ("Нормально", normal_time), ("Высоко", high_time)],
            key=lambda x: x[1]
        )[0]

        task_results.append((task, in_range_pct >= 90, in_range_pct))


        print(f"Задание {idx} ({start:.1f}–{end:.1f} с): цель = {low + PITCH_TOLERANCE:.1f} Гц ±{PITCH_TOLERANCE}")
        print(f"- Попадание в диапазон: {in_range_time:.2f} сек ({in_range_pct:.1f}%)")
        print(f"- Временная маска:")
        print(f"  • Низко:     {low_time:.2f} сек ({low_pct:.1f}%)")
        print(f"  • Нормально: {normal_time:.2f} сек ({normal_pct:.1f}%)")
        print(f"  • Высоко:    {high_time:.2f} сек ({high_pct:.1f}%)")
        print(f"- Итог: попал в зону '{dominant_zone}'")
        print()

        if in_range_pct >= 100:
            successful += 1

    print(f"\n✅ Успешных выполнений: {successful} из {total}")

    # --- Финальный график ---
    fig_final, ax_final = plt.subplots()
    ax_final.set_xlabel("Время (с)")
    ax_final.set_ylabel("Частота (Гц)")
    ax_final.set_title("Итоговая визуализация частоты")
    ax_final.set_xlim(0, GAME_DURATION)
    ax_final.set_ylim(80, 600)
    ax_final.grid()

    ax_final.plot(times, pitches, color="darkblue", label="Сглаженная частота")

    for task, success, accuracy in task_results:
        color = 'green' if success else 'red'
        ax_final.add_patch(Rectangle(
            (task['start'], task['low']),
            TASK_DURATION,
            task['high'] - task['low'],
            facecolor=color,
            alpha=0.2
        ))
        # подпись точности
        ax_final.text(
            x=task['start'] + TASK_DURATION / 2,
            y=task['high'] + 10,
            s=f"{accuracy:.0f}%",
            ha='center',
            va='bottom',
            fontsize=9,
            color=color
        )

    ax_final.legend()
    plt.show()

def main():
    print("🎤 Запуск анализа частоты с заданиями и проверкой тишины...")
    analyze_and_plot_pitch()
    print("🔚 Анализ завершён.")

if __name__ == "__main__":
    main()
