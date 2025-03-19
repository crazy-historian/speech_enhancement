import parselmouth
import matplotlib.pyplot as plt
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
import numpy as np

# Фильтрация резких скачков частоты
def threshold_filter(data, max_jump=50):
    filtered_data = np.array(data)
    for i in range(1, len(filtered_data)):
        if not np.isnan(filtered_data[i]) and not np.isnan(filtered_data[i - 1]):
            if abs(filtered_data[i] - filtered_data[i - 1]) > max_jump:
                filtered_data[i] = filtered_data[i - 1]
    return filtered_data

# Функция скользящего среднего
def moving_average(data, window_size=4):
    smoothed = []
    for i in range(len(data)):
        window = data[max(0, i - window_size + 1):i + 1]  # Берем последние `window_size` блоков
        valid_values = [x for x in window if not np.isnan(x)]  # Исключаем NaN
        smoothed.append(np.nanmean(valid_values) if valid_values else np.nan)  # Если нет валидных значений, оставляем NaN
    return np.array(smoothed)

# Параметры обработки
METHOD = "ac"
PITCH_FLOOR = 100
PITCH_CEILING = 600
TIME_STEP = 0.01
SILENCE_THRESHOLD = 0.03
VOICING_THRESHOLD = 0.45
OCTAVE_COST = 0.01
OCTAVE_JUMP_COST = 0.35
VOICED_UNVOICED_COST = 0.14
BLOCKSIZE = 1024
MAX_JUMP = 50
WINDOW_SIZE = 2 # Размер окна для скользящего среднего

# Создание графиков
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlabel("Время (с)")
ax.set_ylabel("Частота (Гц)")
ax.set_title("График основного тона (скользящее среднее)")
ax.set_ylim(50, 600)
ax.grid()

line_pitch, = ax.plot([], [], color="green", label="Pitch (скользящее среднее)")
plt.legend()
plt.show(block=False)

times = []
pitches = []
current_time = 0

# Запуск потока с микрофона
with InputStream(
    samplerate=16000,  
    blocksize=BLOCKSIZE, 
    channels=1,        
    sampwidth=2        
) as stream:
    print(f"Захват звука с микрофона. Частота дискретизации: {stream.samplerate} Гц")
    stream.set_methods(UnpackRawInFloat32())

    for _ in range(stream.get_iterations(seconds=10)):  
        raw_data = stream.read(BLOCKSIZE)
        if not raw_data:
            continue

        signal = stream.chain_of_methods(raw_data)  
        sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

        # Анализ pitch
        if METHOD == "ac":
            pitch_obj = sound.to_pitch_ac(
                time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                silence_threshold=SILENCE_THRESHOLD, voicing_threshold=VOICING_THRESHOLD,
                octave_cost=OCTAVE_COST, octave_jump_cost=OCTAVE_JUMP_COST,
                voiced_unvoiced_cost=VOICED_UNVOICED_COST
            )
        elif METHOD == "cc":
            pitch_obj = sound.to_pitch_cc(
                time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                silence_threshold=SILENCE_THRESHOLD, voicing_threshold=VOICING_THRESHOLD,
                octave_cost=OCTAVE_COST, octave_jump_cost=OCTAVE_JUMP_COST,
                voiced_unvoiced_cost=VOICED_UNVOICED_COST
            )
        else:
            raise ValueError(f"Неизвестный метод: {METHOD}. Используйте 'ac' или 'cc'.")

        pitch_values = pitch_obj.selected_array['frequency']
        pitch_values[(pitch_values == 0) | (pitch_values > 500)] = np.nan
        avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else np.nan

        # Сохранение данных
        times.append(current_time)
        pitches.append(avg_pitch)

        # Фильтрация выбросов
        #filtered_pitches = threshold_filter(pitches, max_jump=MAX_JUMP)

        # Применение скользящего среднего
        smoothed_pitches = moving_average(pitches, WINDOW_SIZE)

        # Вывод в консоль
        print(f"{current_time:.2f} s | Pitch: {avg_pitch:.2f} Hz | Smoothed: {smoothed_pitches[-1]:.2f} Hz")

        # Обновление графика
        ax.cla()
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Частота (Гц)")
        ax.set_title("График основного тона (скользящее среднее)")
        ax.set_ylim(100, 900)
        ax.grid()
        ax.plot(times, smoothed_pitches, color="green", label="Pitch (скользящее среднее)")
        plt.legend()
        plt.draw()
        plt.pause(0.01)

        current_time += BLOCKSIZE / stream.samplerate  

plt.show(block=True)
