import parselmouth
import matplotlib.pyplot as plt
from audiochains.streams import StreamFromFile
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

def process_audio(
    audio_file,
    method="ac",
    pitch_floor=75,
    pitch_ceiling=600,
    time_step=0.01,
    silence_threshold_db=-25.0,
    silence_threshold=0.04,
    voicing_threshold=0.45,
    octave_cost=0.01,
    octave_jump_cost=0.35,
    voiced_unvoiced_cost=0.14,
    blocksize=1024,
    max_jump=50,
    blocks_to_silent=2
):
    times = []  
    intensities = []  
    pitches = []  
    activity_states = []  

    current_state = 0
    silent_counter = 0  

    with StreamFromFile(filename=audio_file, blocksize=blocksize) as stream:
        print(f"Обработка аудиофайла {audio_file}. Частота дискретизации: {stream.samplerate} Гц")
        stream.set_methods(UnpackRawInFloat32())

        current_time = 0  

        for i in range(stream.get_iterations()):
            raw_data = stream.read(blocksize)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)  
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

            if method == "ac":
                pitch_obj = sound.to_pitch_ac(
                    time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling, silence_threshold=silence_threshold,
                    voicing_threshold=voicing_threshold, octave_cost=octave_cost,
                    octave_jump_cost=octave_jump_cost, voiced_unvoiced_cost=voiced_unvoiced_cost
                )
            elif method == "cc":
                pitch_obj = sound.to_pitch_cc(
                    time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling,
                    voicing_threshold=voicing_threshold, octave_cost=octave_cost,
                    octave_jump_cost=octave_jump_cost, voiced_unvoiced_cost=voiced_unvoiced_cost
                )
            else:
                raise ValueError(f"Неизвестный метод: {method}. Используйте 'ac' или 'cc'.")

            pitch_values = pitch_obj.selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > 500)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else np.nan
            
            
            above_silence_threshold = avg_intensity > silence_threshold_db
            valid_pitch = avg_pitch > pitch_floor

            if above_silence_threshold and valid_pitch:
                current_state = 1
                silent_counter = 0
            else:
                silent_counter += 1
                if silent_counter >= blocks_to_silent:
                    current_state = 0

            times.append(current_time)
            intensities.append(avg_intensity)
            pitches.append(avg_pitch)
            activity_states.append(current_state)

            print(f"{current_time:.2f} s - {'Voiced (1)' if current_state else 'Silent (0)'} | "
                  f"Intensity: {avg_intensity:.2f} dB | Pitch: {avg_pitch:.2f} Hz")
            print(f"   Pitch values: {pitch_values}")  # Вывод всех значений pitch в блоке
            print(f"   Intensity values: {intensity_values}")  # Вывод всех значений интенсивности в блоке")

            current_time += blocksize / stream.samplerate

    #filtered_pitches = threshold_filter(pitches, max_jump=max_jump)

    print(f"Количество точек pitch: {len(pitches)}")
    print(f"Минимальный pitch: {np.nanmin(pitches) if len(pitches) > 0 else 'Нет данных'}")
    print(f"Максимальный pitch: {np.nanmax(pitches) if len(pitches) > 0 else 'Нет данных'}'")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(times, intensities, color="blue")
    axes[0].set_ylabel("Интенсивность (дБ)")
    axes[0].set_title("Интенсивность звука")

    if len(pitches) > 0 and np.any(~np.isnan(pitches)):
        axes[1].plot(times, pitches, color="red")
        axes[1].scatter(times, pitches, color="black", s=5)  
        axes[1].set_ylabel("Частота (Гц)")
        axes[1].set_title("Частота основного тона")
    else:
        axes[1].text(0.5, 0.5, "Нет данных для pitch", ha="center", va="center", fontsize=12)
        axes[1].set_ylabel("Частота (Гц)")
        axes[1].set_title("Частота основного тона")

    axes[2].step(times, activity_states, color="green", where="post")
    axes[2].set_ylabel("Активность (0 - silent, 1 - voiced)")
    axes[2].set_xlabel("Время (с)")
    axes[2].set_title("Активность речи")

    plt.tight_layout()
    plt.show()

    return times, pitches, activity_states

process_audio(
    audio_file="silero_vad/files/testnaslogi2.wav",
    method="ac",
    pitch_floor=100,
    pitch_ceiling=600,
    time_step=0.01,
    silence_threshold_db=-25.0,
    silence_threshold=0.04,
    voicing_threshold=0.45,
    octave_cost=0.01,
    octave_jump_cost=0.35,
    voiced_unvoiced_cost=0.14,
    blocksize=1024,
    max_jump=20,
    blocks_to_silent=2
)
