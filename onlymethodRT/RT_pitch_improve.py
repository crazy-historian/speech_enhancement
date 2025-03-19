import time
import numpy as np
import parselmouth
import matplotlib.pyplot as plt
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

BLOCKSIZE = 1024
PITCH_FLOOR = 100
PITCH_CEILING = 600
VOICING_THRESHOLD = 0.6
SILENCE_THRESHOLD_DB = 45.0
BLOCKS_TO_SILENT = 6
GAME_DURATION = 10  # Длительность измерения частоты

def main():
    """
    Главная функция для анализа частоты и построения графика в реальном времени.
    """
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Частота (Гц)")
        ax.set_title("График основного тона в реальном времени")
        ax.set_ylim(PITCH_FLOOR, PITCH_CEILING)
        ax.grid()

        line_pitch, = ax.plot([], [], color="green", label="Частота (Гц)")
        plt.legend()
        plt.show(block=False)

        times = []
        pitches = []
        start_time = time.time()
        silent_counter = 0
        last_valid_pitch = None
        block_times = []

        while time.time() - start_time < GAME_DURATION:
            block_start_time = time.time()
            
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            # Извлечение интенсивности
            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

            # Анализ частоты (pitch)
            pitch_obj = sound.to_pitch_ac(
                time_step=0.01,
                pitch_floor=PITCH_FLOOR,
                pitch_ceiling=PITCH_CEILING,
                voicing_threshold=VOICING_THRESHOLD
            )
            pitch_values = pitch_obj.selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else None

            # Условие для отображения частоты только при достаточной громкости
            if avg_intensity > SILENCE_THRESHOLD_DB:
                silent_counter = 0
                last_valid_pitch = avg_pitch
            else:
                silent_counter += 1
                if silent_counter >= BLOCKS_TO_SILENT:
                    last_valid_pitch = None  # Прекращаем рисовать после BLOCKS_TO_SILENT тишины

            current_time = time.time() - start_time
            times.append(current_time)
            pitches.append(last_valid_pitch)

            # Очистка графика перед обновлением
            ax.cla()
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Частота (Гц)")
            ax.set_title("График основного тона в реальном времени")
            ax.set_ylim(PITCH_FLOOR, PITCH_CEILING)
            ax.grid()
            ax.plot(times, pitches, color="green", label="Частота (Гц)")
            plt.legend()

            plt.draw()
            plt.pause(0.01)
            
            block_end_time = time.time()
            block_processing_time = block_end_time - block_start_time
            block_times.append(block_processing_time)
            print(f"Обработка блока заняла: {block_processing_time:.5f} секунд")
    
    avg_block_time = np.mean(block_times)
    print(f"Среднее время обработки блока: {avg_block_time:.5f} секунд")
    plt.show()
    print("Анализ завершён.")

if __name__ == "__main__":
    main()