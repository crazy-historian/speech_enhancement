import parselmouth
from audiochains.streams import StreamFromFile
from audiochains.block_methods import UnpackRawInFloat32
import numpy as np
import matplotlib.pyplot as plt

# Параметры порогов
SILENCE_THRESHOLD_DB = -10.0  # Порог громкости (дБ)
PITCH_FLOOR = 100  # Минимальная частота (Гц)
BLOCKS_TO_SILENT = 2  # Количество блоков без звука перед сменой состояния на silent

# Файл для обработки
AUDIO_FILE = "silero_vad/files/testnaslogi2.wav"
BLOCKSIZE = 1024  # Размер блока

# Переменные для хранения результатов
times = []  # Временные метки
intensities = []  # Интенсивность (дБ)
pitches = []  # Частота основного тона (Гц)
activity_states = []  # 0 - silent, 1 - voiced

# Переменная состояния (0 - silent, 1 - voiced)
current_state = 0
silent_counter = 0  # Количество блоков без звука

# Открываем текстовый файл для записи состояний
with open("state_log.txt", "w") as log_file:
    with StreamFromFile(filename=AUDIO_FILE, blocksize=BLOCKSIZE) as stream:
        print(f"Обработка аудиофайла {AUDIO_FILE}. Частота дискретизации: {stream.samplerate} Гц")
        stream.set_methods(UnpackRawInFloat32())  # Декодируем поток в float32

        current_time = 0  # Время начала текущего блока

        for i in range(stream.get_iterations()):
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)  # Преобразуем в float32
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            # Вычисляем интенсивность
            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()

            # Анализ pitch (основного тона)
            pitch_obj = sound.to_pitch()
            pitch_values = pitch_obj.selected_array['frequency']

            # Средние значения для блока
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50
            avg_pitch = np.mean([p for p in pitch_values if p > 0]) if np.any(pitch_values > 0) else 0

            # Проверяем условия
            above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
            valid_pitch = avg_pitch > PITCH_FLOOR

            # Определяем состояние
            if above_silence_threshold and valid_pitch:
                if current_state == 0:
                    log_file.write(f"{current_time:.2f} - Voiced (1)\n")
                current_state = 1
                silent_counter = 0
            else:
                silent_counter += 1
                if silent_counter >= BLOCKS_TO_SILENT and current_state == 1:
                    log_file.write(f"{current_time:.2f} - Silent (0)\n")
                    current_state = 0

            # Сохраняем данные для графиков
            times.append(current_time)
            intensities.append(avg_intensity)
            pitches.append(avg_pitch)
            activity_states.append(current_state)

            # Вывод состояния в консоль
            print(f"{current_time:.2f} s - {'Voiced (1)' if current_state else 'Silent (0)'}")

            # Переход к следующему временному блоку
            current_time += BLOCKSIZE / stream.samplerate

print("Обработка завершена. Состояния сохранены в state_log.txt.")

# Построение графиков
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# График интенсивности
axes[0].plot(times, intensities, color="blue")
axes[0].set_ylabel("Интенсивность (дБ)")
axes[0].set_title("Интенсивность звука")

# График частоты
axes[1].plot(times, pitches, color="red")
axes[1].set_ylabel("Частота (Гц)")
axes[1].set_title("Частота основного тона")

# График активности речи
axes[2].step(times, activity_states, color="green", where="post")
axes[2].set_ylabel("Активность (0 - silent, 1 - voiced)")
axes[2].set_xlabel("Время (с)")
axes[2].set_title("Активность речи")

plt.tight_layout()
plt.show()
