import time
import numpy as np
import random
import parselmouth
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

BLOCKSIZE = 1024
GAME_DURATION = 60
BURST_INTERVAL = 6
BURST_DURATION = 2
FIRST_BURST_DELAY = 6
INTENSITY_MIN = 60
INTENSITY_MAX = 61
INTENSITY_TOLERANCE = 10  # вверх/вниз от цели

def generate_tasks():
    tasks = []
    t = FIRST_BURST_DELAY
    while t + BURST_DURATION <= GAME_DURATION:
        target = random.uniform(INTENSITY_MIN, INTENSITY_MAX)
        tasks.append({
            'start': t,
            'end': t + BURST_DURATION,
            'min_db': target - INTENSITY_TOLERANCE,
            'max_db': target + INTENSITY_TOLERANCE,
            'target': target
        })
        t += BURST_INTERVAL
    return tasks

def analyze_and_plot_voice_intensity():
    tasks = generate_tasks()

    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())

        fig, ax = plt.subplots()

        times = []
        intensities = []
        processing_times = []
        start_time = time.time()
        current_time = 0
        WINDOW_WIDTH = 12

        while time.time() - start_time < GAME_DURATION:
            block_start_time = time.time()

            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)
            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else 0

            times.append(current_time)
            intensities.append(avg_intensity)

            # Полная перерисовка
            ax.cla()
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Интенсивность (dB)")
            ax.set_title("Интенсивность в реальном времени")
            ax.set_xlim(max(0, current_time - (WINDOW_WIDTH - 2)), current_time + 2)
            ax.set_ylim(0, 100)
            ax.grid()

            # Отображение задания как прямоугольников
            for task in tasks:
                if task['start'] <= current_time + 2:
                    rect = Rectangle(
                        (task['start'], task['min_db']),
                        BURST_DURATION,
                        task['max_db'] - task['min_db'],
                        linewidth=1,
                        edgecolor='gray',
                        facecolor='gray',
                        alpha=0.2
                    )
                    ax.add_patch(rect)

            # Интенсивность
            ax.plot(times, intensities, color="red", label="Интенсивность (dB)")
            ax.legend()

            plt.pause(0.01)

            block_end_time = time.time()
            processing_times.append(block_end_time - block_start_time)

            current_time += BLOCKSIZE / 16000

        avg_processing_time = np.mean(processing_times) if processing_times else 0
        print(f"\nСреднее время обработки блока: {avg_processing_time:.6f} секунд")

        # ✅ Проверка попаданий
        hits = 0
        for task in tasks:
            indices = [i for i, t in enumerate(times) if task['start'] <= t < task['end']]
            values = [intensities[i] for i in indices]
            if values and all(task['min_db'] <= v <= task['max_db'] for v in values):
                task['hit'] = True
                hits += 1
            else:
                task['hit'] = False

        # 🎬 Финальный график
        print(f"\n🎯 Удачных попаданий: {hits} из {len(tasks)}")

        fig_final, ax_final = plt.subplots()
        ax_final.set_xlabel("Время (с)")
        ax_final.set_ylabel("Интенсивность (dB)")
        ax_final.set_title("Итоговая визуализация интенсивности")
        ax_final.set_xlim(0, GAME_DURATION)
        ax_final.set_ylim(0, 100)
        ax_final.grid()
        ax_final.plot(times, intensities, color="darkred", label="Интенсивность (dB)")

        # Отображение заданий (успех/неуспех)
        for task in tasks:
            color = 'green' if task.get('hit') else 'red'
            rect = Rectangle(
                (task['start'], task['min_db']),
                BURST_DURATION,
                task['max_db'] - task['min_db'],
                linewidth=0,
                edgecolor=None,
                facecolor=color,
                alpha=0.3
            )
            ax_final.add_patch(rect)

        ax_final.legend()
        plt.show()

def main():
    print("🚀 Запуск анализа голосовой интенсивности с заданиями...")
    analyze_and_plot_voice_intensity()
    print("✅ Анализ завершён.")

if __name__ == "__main__":
    main()
