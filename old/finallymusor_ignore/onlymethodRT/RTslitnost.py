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
BLOCKS_TO_SILENT = 1
GAME_DURATION = 10  # Длительность измерения частоты

def analyze_voice():
    """
    Функция анализирует голос и строит бинарный график (0 - тишина, 1 - голос).
    """
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Голосовой сигнал")
        ax.set_title("Анализ слитности и раздельности речи")
        ax.set_ylim(-0.1, 1.1)
        ax.grid()

        line_voice, = ax.plot([], [], color="blue", label="Голос (1 - есть, 0 - нет)")
        plt.legend()
        plt.show(block=False)

        times = []
        voice_states = []
        start_time = time.time()
        silent_counter = 0
        block_times = []
        last_voice_state = 0

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
            
            last_voice_state = voice_state

            current_time = time.time() - start_time
            times.append(current_time)
            voice_states.append(voice_state)

            # Очистка графика перед обновлением
            ax.cla()
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Голосовой сигнал")
            ax.set_title("Анализ слитности и раздельности речи")
            ax.set_ylim(-0.1, 1.1)
            ax.grid()
            ax.plot(times, voice_states, color="blue", label="Голос (1 - есть, 0 - нет)")
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
    analyze_voice()