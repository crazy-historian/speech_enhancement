import parselmouth
import matplotlib.pyplot as plt
from audiochains.streams import StreamFromFile
from audiochains.block_methods import UnpackRawInFloat32
import numpy as np

audio_file = 'silero_vad/files/female_22.wav'
praat_file = 'silero_vad/files/praat2.txt'
threshold = 0.5
blocksize = 1024
sampwidth = 2
pitch_floor = 75.0

times = []
frequencies = []

# Чтение значений из praat2.txt
praat_times = []
praat_frequencies = []
try:
    with open(praat_file, 'r') as f:
        next(f)  # Пропускаем заголовок
        for line in f:
            time, freq = line.strip().split()
            time = float(time)
            freq = float(freq) if freq != '--undefined--' else np.nan
            praat_times.append(time)
            praat_frequencies.append(freq)
except Exception as e:
    print(f"Ошибка при чтении файла {praat_file}: {e}")

with StreamFromFile(
    filename=audio_file,
    blocksize=blocksize
) as stream:

    print(f"Частота дискретизации: {stream.samplerate} Гц")

    stream.set_methods(UnpackRawInFloat32())

    current_time = 0
    for i in range(stream.get_iterations()):
        try:

            raw_data = stream.read(blocksize)


            if not raw_data:
                print(f"Блок {i} пустой, пропускаем...")
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            pitch = sound.to_pitch(pitch_floor=pitch_floor)
            frequency = pitch.selected_array['frequency']
            frequency[frequency == 0] = np.nan

            times.extend(current_time + pitch.xs())
            frequencies.extend(frequency)

            current_time += blocksize / stream.samplerate

        except Exception as err:
            print(f"Ошибка при обработке блока {i}: {err}")

if times and frequencies:

    plt.figure(figsize=(12, 8))

    plt.plot(times, frequencies, label="Основной тон (parselmouth)", color="blue")

    if praat_times and praat_frequencies:
        plt.plot(praat_times, praat_frequencies, label="Основной тон (Praat)", color="red", linestyle="--")

    plt.xlabel("Время (с)")
    plt.ylabel("Частота (Гц)")
    plt.title("График основного тона")
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Данные для построения графика отсутствуют.")
