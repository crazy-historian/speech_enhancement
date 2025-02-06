import librosa
import numpy as np
import matplotlib.pyplot as plt
from audiochains.streams import StreamFromFile
from audiochains.block_methods import UnpackRawInFloat32
import time as t

def process_audio_with_pyin(
    audio_file,                  # Путь к аудиофайлу
    praat_file=None,             # Путь к файлу Praat
    pitch_floor=75.0,            # Минимальная частота (Гц)
    pitch_ceiling=600.0,         # Максимальная частота (Гц)
    time_step=0.01,              # Шаг анализа (секунды)
    blocksize=2048,              # Размер блока
    plot_results=True            # Построить график
):
    """
    Функция для обработки аудио с использованием метода pYIN из librosa.
    """
    # Локальные переменные для результатов
    times = []
    frequencies = []
    praat_times = []
    praat_frequencies = []
    processing_times = []

    # Если указан файл Praat, читаем его данные
    if praat_file:
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

    # Обработка аудиофайла с использованием StreamFromFile
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

                # Измеряем время начала обработки
                start_time = t.time()

                # Преобразуем данные в массив NumPy
                signal = stream.chain_of_methods(raw_data)

                # Вычисление высоты тона с помощью pYIN
                frame_length = int(time_step * stream.samplerate)  # Размер кадра в сэмплах
                hop_length = frame_length // 2                    # Шаг между кадрами
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    signal,
                    fmin=pitch_floor,
                    fmax=pitch_ceiling,
                    sr=stream.samplerate,
                    frame_length=frame_length,
                    hop_length=hop_length,
                    fill_na=np.nan  # Заполняем неозвученные участки NaN
                )

                # Временные метки для каждого кадра
                block_times = librosa.frames_to_time(
                    np.arange(len(f0)),
                    sr=stream.samplerate,
                    hop_length=hop_length
                )
                block_times += current_time  # Сдвиг времени на текущий блок

                # Добавляем результаты в общие списки
                times.extend(block_times)
                frequencies.extend(f0)

                # Обновляем текущее время
                current_time += blocksize / stream.samplerate

                # Измеряем время окончания обработки и рассчитываем разницу
                end_time = t.time()
                processing_times.append(end_time - start_time)

            except Exception as err:
                print(f"Ошибка при обработке блока {i}: {err}")

    # Вычисление среднего времени обработки одного блока
    if processing_times:
        avg_processing_time = sum(processing_times) / len(processing_times)
        print(f"Среднее время обработки одного блока ({blocksize} сэмплов): {avg_processing_time:.6f} секунд")
    else:
        print("Обработка не производилась, данные отсутствуют.")

    # Построение графика
    if plot_results and times and frequencies:
        plt.figure(figsize=(12, 8))

        # График для pYIN
        plt.plot(times, frequencies, label="Основной тон (pYIN)", color="blue")

        # График для файла Praat (если есть данные)
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

    # Возврат результатов
    return times, frequencies, praat_times, praat_frequencies

# Пути к файлам
audio1 = 'silero_vad/files/femalechist2.wav'  # Женский голос без шумов
praat1 = 'silero_vad/files/praatfemalechist.txt'  # Женский голос без шумов

# Вызов функции
process_audio_with_pyin(
    audio1,                  # Путь к аудиофайлу
    praat_file=praat1,       # Путь к файлу Praat
    pitch_floor=75.0,        # Минимальная частота (Гц)
    pitch_ceiling=600.0,     # Максимальная частота (Гц)
    time_step=0.01,          # Шаг анализа (секунды)
    blocksize=1024,          # Размер блока
    plot_results=True        # Построить график
)