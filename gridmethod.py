import parselmouth
import matplotlib.pyplot as plt
import numpy as np

def detect_pauses_and_create_textgrid(
    audio_file,
    output_textgrid_file="output.TextGrid",
    silence_threshold=-25.0,
    min_silence_duration=0.2
):
    sound = parselmouth.Sound(audio_file)

    intensity = sound.to_intensity()
    time_points = intensity.xs()
    intensity_values = intensity.values.T.flatten()

    is_silence = intensity_values < silence_threshold
    silence_intervals = []
    start_time = None

    for i, silent in enumerate(is_silence):
        time = time_points[i]
        if silent and start_time is None:
            start_time = time
        elif not silent and start_time is not None:
            end_time = time
            if end_time - start_time >= min_silence_duration:
                silence_intervals.append((start_time, end_time))
            start_time = None

    if start_time is not None:
        silence_intervals.append((start_time, time_points[-1]))

    # Создание TextGrid файла
    from parselmouth.praat import call

    textgrid = call("Create TextGrid...", 0, sound.get_total_duration(), "Pauses", "")
    for start, end in silence_intervals:
        call(textgrid, "Insert interval tier", 1, "Pauses")
        call(textgrid, "Insert interval", 1, start, end, "pause")

    call(textgrid, "Save as short text file...", output_textgrid_file)
    print(f"TextGrid файл создан: {output_textgrid_file}")

    # Построение графика
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, intensity_values, label="Интенсивность", color="blue")

    for start, end in silence_intervals:
        plt.axvspan(start, end, color="red", alpha=0.3, label="Пауза" if silence_intervals.index((start, end)) == 0 else None)

    plt.xlabel("Время (с)")
    plt.ylabel("Интенсивность (дБ)")
    plt.title("Интенсивность и паузы")
    plt.legend()
    plt.grid()
    plt.show()

# Пример использования
if __name__ == "__main__":
    audio_file = "silero_vad/files/testTextGrid1.wav"
    output_textgrid_file = "output.TextGrid"
    detect_pauses_and_create_textgrid(
        audio_file,
        output_textgrid_file=output_textgrid_file,
        silence_threshold=-15.0,
        min_silence_duration=0.2
    )
