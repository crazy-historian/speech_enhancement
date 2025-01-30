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

# Параметры обработки
METHOD = "ac"
PITCH_FLOOR = 100
PITCH_CEILING = 600
TIME_STEP = 0.01
SILENCE_THRESHOLD_DB = -25.0
VOICING_THRESHOLD = 0.45
OCTAVE_COST = 0.01
OCTAVE_JUMP_COST = 0.35
VOICED_UNVOICED_COST = 0.14
BLOCKSIZE = 1024
MAX_JUMP = 50
BLOCKS_TO_SILENT = 2

# Подготовка графиков
plt.ion()
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

# Фиксированные лимиты осей
axes[0].set_ylim(0, 120)   # Интенсивность звука (дБ)
axes[1].set_ylim(50, 600)  # Частота (Гц)
axes[2].set_ylim(-0.5, 1.5) # Активность (0 или 1)

plt.show(block=False)

times = []
intensities = []
pitches = []
activity_states = []
current_time = 0
current_state = 0
silent_counter = 0  

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

        # Анализ громкости (интенсивности)
        intensity_obj = sound.to_intensity()
        intensity_values = intensity_obj.values.T.flatten()
        avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

        # Анализ pitch
        if METHOD == "ac":
            pitch_obj = sound.to_pitch_ac(
                time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                voicing_threshold=VOICING_THRESHOLD, octave_cost=OCTAVE_COST,
                octave_jump_cost=OCTAVE_JUMP_COST, voiced_unvoiced_cost=VOICED_UNVOICED_COST
            )
        elif METHOD == "cc":
            pitch_obj = sound.to_pitch_cc(
                time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING,
                voicing_threshold=VOICING_THRESHOLD, octave_cost=OCTAVE_COST,
                octave_jump_cost=OCTAVE_JUMP_COST, voiced_unvoiced_cost=VOICED_UNVOICED_COST
            )
        else:
            raise ValueError(f"Неизвестный метод: {METHOD}. Используйте 'ac' или 'cc'.")

        pitch_values = pitch_obj.selected_array['frequency']
        pitch_values[(pitch_values == 0) | (pitch_values > 500)] = np.nan
        avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else np.nan

        # Определение состояния (Voiced/Silent)
        above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
        valid_pitch = avg_pitch > PITCH_FLOOR

        if above_silence_threshold and valid_pitch:
            current_state = 1
            silent_counter = 0
        else:
            silent_counter += 1
            if silent_counter >= BLOCKS_TO_SILENT:
                current_state = 0

        # Сохранение данных
        times.append(current_time)
        intensities.append(avg_intensity)
        pitches.append(avg_pitch)
        activity_states.append(current_state)

        # Фильтрация выбросов
        filtered_pitches = threshold_filter(pitches, max_jump=MAX_JUMP)

        # Вывод в консоль
        print(f"{current_time:.2f} s - {'Voiced (1)' if current_state else 'Silent (0)'} | "
              f"Intensity: {avg_intensity:.2f} dB | Pitch: {avg_pitch:.2f} Hz")

        # Обновление графиков
        axes[0].cla()
        axes[0].plot(times, intensities, color="blue")
        axes[0].set_ylabel("Интенсивность (дБ)")
        axes[0].set_title("Интенсивность звука")
        axes[0].set_ylim(0, 120)  

        axes[1].cla()
        axes[1].plot(times, filtered_pitches, color="red")
        axes[1].scatter(times, filtered_pitches, color="black", s=5)  
        axes[1].set_ylabel("Частота (Гц)")
        axes[1].set_title("Частота основного тона")
        axes[1].set_ylim(50, 600)  

        axes[2].cla()
        axes[2].step(times, activity_states, color="green", where="post")
        axes[2].set_ylabel("Активность (0 - silent, 1 - voiced)")
        axes[2].set_xlabel("Время (с)")
        axes[2].set_title("Активность речи")
        axes[2].set_ylim(-0.5, 1.5)  

        plt.draw()  # Обновляем график
        plt.pause(0.05)  # Обновляем с задержкой

        current_time += BLOCKSIZE / stream.samplerate  
