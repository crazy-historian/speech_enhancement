import parselmouth
from parselmouth import TextGrid
from parselmouth.praat import call
import matplotlib.pyplot as plt
import numpy as np
from audiochains.streams import StreamFromFile
from audiochains.block_methods import UnpackRawInFloat32
import os
def detect_pauses_and_create_textgrid_realtime(
    audio_file,
    output_textgrid_file="output.TextGrid",
    silence_threshold=-25.0,  # Порог громкости для определения тишины (в децибелах)
    min_silence_duration=0.2,  # Минимальная длительность паузы (в секундах)
    blocksize=1024
):
    """
    Создает TextGrid-аннотацию пауз на основе аудиофайла, обрабатываемого в реальном времени.

    :param audio_file: Путь к аудиофайлу.
    :param output_textgrid_file: Путь к выходному файлу TextGrid.
    :param silence_threshold: Порог громкости для определения тишины (в децибелах).
    :param min_silence_duration: Минимальная длительность паузы (в секундах).
    :param blocksize: Размер блока для обработки в реальном времени.
    """
    # Локальные переменные для пауз
    silence_intervals = []
    start_time = None
    current_time = 0

    # Обработка звукового файла
    with StreamFromFile(filename=audio_file, blocksize=blocksize) as stream:
        print(f"Частота дискретизации: {stream.samplerate} Гц")
        stream.set_methods(UnpackRawInFloat32())

        for i in range(stream.get_iterations()):
            try:
                raw_data = stream.read(blocksize)
                if not raw_data:
                    continue

                signal = stream.chain_of_methods(raw_data)
                sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

                # Вычисляем интенсивность
                intensity = sound.to_intensity()
                time_points = intensity.xs() + current_time
                intensity_values = intensity.values.T.flatten()

                # Определение участков тишины
                is_silence = intensity_values < silence_threshold

                for j, silent in enumerate(is_silence):
                    time = time_points[j]
                    if silent and start_time is None:
                        start_time = time
                    elif not silent and start_time is not None:
                        end_time = time
                        if end_time - start_time >= min_silence_duration:
                            silence_intervals.append((start_time, end_time))
                        start_time = None

                # Добавление последней паузы, если блок заканчивается на тишине
                if start_time is not None and i == stream.get_iterations() - 1:
                    silence_intervals.append((start_time, time_points[-1]))

                current_time += blocksize / stream.samplerate

            except Exception as err:
                print(f"Ошибка при обработке блока {i}: {err}")

    
    # Создание TextGrid файла
    textgrid = TextGrid(0, current_time)  # Указываем начальное и конечное время
    silence_tier = textgrid.add_tier("Pauses")

    for start, end in silence_intervals:
        silence_tier.add_interval(start, end, "pause")

    textgrid.save(output_textgrid_file, "short")

    print(f"TextGrid файл создан: {output_textgrid_file}")

    # Построение графика интенсивности с паузами
    plt.figure(figsize=(12, 6))
    for start, end in silence_intervals:
        plt.axvspan(start, end, color="red", alpha=0.3, label="Пауза" if silence_intervals.index((start, end)) == 0 else None)

    plt.xlabel("Время (с)")
    plt.ylabel("Интенсивность (дБ)")
    plt.title("Интенсивность и паузы (реальное время)")
    plt.legend()
    plt.grid()
    plt.show()

# Пример использования
if __name__ == "__main__":
    audio_file = "silero_vad/files/test1nonoise.wav"  # Укажите путь к вашему аудиофайлу
    output_textgrid_file = "output.TextGrid"
    detect_pauses_and_create_textgrid_realtime(
        audio_file,
        output_textgrid_file=output_textgrid_file,
        silence_threshold=-25.0,
        min_silence_duration=0.2,
        blocksize=1024
    )
