import time
import numpy as np
import parselmouth
import matplotlib.pyplot as plt
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

BLOCKSIZE = 1024
SILENCE_THRESHOLD_DB = 40.0
GAME_DURATION = 10  # Длительность измерения интенсивности

def analyze_and_plot_voice_intensity():
    """
    Анализирует голосовую интенсивность и строит график в реальном времени.
    """
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())

        fig, ax = plt.subplots()
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Интенсивность (dB)")
        ax.set_title("График интенсивности")
        ax.set_xlim(0, GAME_DURATION)
        ax.set_ylim(0, 100)
        ax.grid()

        line_intensity, = ax.plot([], [], color="red", label="Интенсивность (dB)")
        plt.legend()
        plt.show(block=False)

        times = []
        intensities = []
        processing_times = []  # Хранение времени обработки блоков
        start_time = time.time()
        current_time = 0

        while time.time() - start_time < GAME_DURATION:
            block_start_time = time.time()  # Засекаем время начала обработки блока
            
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            # Преобразование аудиоданных
            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            # Извлечение интенсивности
            intensity_obj = sound.to_intensity(subtract_mean=False)
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else 0
            print(intensity_obj)
            for t, val in zip(intensity_obj.xs().flatten(), intensity_values):
                print(f"t={t:.3f}s: {val:.2f} dB")
            print("Intensity values:", np.round(intensity_values, 2))
            print("Average intensity:", round(avg_intensity, 2))

            # Добавление новых данных
            times.append(current_time)
            intensities.append(avg_intensity)

            # Обновление графика
            line_intensity.set_xdata(times)
            line_intensity.set_ydata(intensities)
            ax.set_xlim(0, max(GAME_DURATION, current_time + 1))
            ax.set_ylim(0, 100)

            plt.draw()
            plt.pause(0.01)

            # Вывод времени обработки блока
            block_end_time = time.time()
            block_processing_time = block_end_time - block_start_time
            processing_times.append(block_processing_time)
            print(f"Обработка блока {BLOCKSIZE} заняла {block_processing_time:.6f} секунд")

            current_time += BLOCKSIZE / 16000  # Обновление времени
        
        plt.show()

        # Вывод среднего времени обработки
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        print(f"Среднее время обработки одного блока: {avg_processing_time:.6f} секунд")

def main():
    """
    Главная функция для запуска анализа голосовой интенсивности и построения графика.
    """
    print("Запуск анализа голосовой интенсивности...")
    analyze_and_plot_voice_intensity()
    print("Анализ завершён.")

if __name__ == "__main__":
    main()
