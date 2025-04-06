import time
import numpy as np
import parselmouth
import matplotlib.pyplot as plt
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
import wave
import struct

BLOCKSIZE = 1024
PITCH_FLOOR = 100
PITCH_CEILING = 600
VOICING_THRESHOLD = 0.6
SILENCE_THRESHOLD_DB = 45.0
BLOCKS_TO_SILENT = 2
GAME_DURATION = 60  # сек
BURST_INTERVAL = 6
BURST_DURATION = 2
FIRST_BURST_DELAY = 6

def generate_task_line(duration: int, burst_interval: int = 6, burst_duration: int = 2, first_burst_delay: int = 6):
    step = 0.1
    time_points = np.arange(0, duration, step)
    task_states = np.zeros_like(time_points)
    for i, t in enumerate(time_points):
        if t >= first_burst_delay and (t - first_burst_delay) % burst_interval < burst_duration:
            task_states[i] = 1
    return time_points, task_states

def count_successful_hits(task_times, task_states, voice_times, voice_states,
                          burst_interval=6, burst_duration=2, first_burst_delay=6):
    step = 0.1
    successful_hits = 0
    total_tasks = 0
    task_hits = []

    t = first_burst_delay
    while t + burst_duration <= GAME_DURATION:
        total_tasks += 1
        indices = [i for i, time in enumerate(voice_times) if t <= time < t + burst_duration]
        if indices and all(voice_states[i] == 1 for i in indices):
            successful_hits += 1
            task_hits.append((t, t + burst_duration, True))
        else:
            task_hits.append((t, t + burst_duration, False))
        t += burst_interval

    return successful_hits, total_tasks, task_hits

def analyze_voice():
    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
        stream.set_methods(UnpackRawInFloat32())

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Состояние")
        ax.set_title("Анализ слитности и раздельности речи")
        ax.set_ylim(-0.1, 1.1)
        ax.grid()

        task_times, task_states = generate_task_line(GAME_DURATION, BURST_INTERVAL, BURST_DURATION, FIRST_BURST_DELAY)
        ax.plot(task_times, task_states, color="gray", linestyle='-', linewidth=2, label="Задание (цель)")

        line_voice, = ax.plot([], [], color="blue", linestyle='--', label="Голос (факт)")
        plt.legend()
        plt.show(block=False)

        times = []
        voice_states = []
        start_time = time.time()
        silent_counter = 0
        last_voice_state = 0
        block_times = []
        raw_audio_data = []

        WINDOW_WIDTH = 12

        while time.time() - start_time < GAME_DURATION:
            block_start_time = time.time()

            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue
            raw_audio_data.append(raw_data)

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

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

            line_voice.set_data(times, voice_states)
            ax.set_xlim(max(0, current_time - (WINDOW_WIDTH - 2)), current_time + 2)
            plt.draw()
            plt.pause(0.01)

            block_end_time = time.time()
            block_times.append(block_end_time - block_start_time)

        # Анализ попаданий
        successful_hits, total_tasks, task_hit_intervals = count_successful_hits(
            task_times, task_states, times, voice_states,
            burst_interval=BURST_INTERVAL,
            burst_duration=BURST_DURATION,
            first_burst_delay=FIRST_BURST_DELAY
        )

        print(f"\n✅ Удачных попаданий: {successful_hits} из {total_tasks}")
        print(f"🎯 Точность: {successful_hits / total_tasks * 100:.1f}%")

        # Финальный график
        ax.cla()
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Состояние")
        ax.set_title("Итоговый анализ речи с попаданиями")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, GAME_DURATION)
        ax.grid()
        ax.plot(task_times, task_states, color="gray", linestyle='-', linewidth=2, label="Задание (цель)")
        ax.plot(times, voice_states, color="blue", linestyle='--', label="Голос (факт)")

        for start, end, success in task_hit_intervals:
            ax.axvspan(start, end, color='green' if success else 'red', alpha=0.2)

        plt.legend()
        plt.show()
        # Сохраняем запись
        output_filename = "voice_recording.wav"
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b''.join(raw_audio_data))

        print(f"💾 Запись голоса сохранена в файл: {output_filename}")

        avg_block_time = np.mean(block_times)
        print(f"⏱ Среднее время обработки блока: {avg_block_time:.5f} секунд")
        print("🧠 Анализ завершён.")

if __name__ == "__main__":
    analyze_voice()
