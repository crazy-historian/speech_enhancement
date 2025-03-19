import parselmouth
import matplotlib.pyplot as plt
from audiochains.streams import StreamFromFile
from audiochains.block_methods import UnpackRawInFloat32
import numpy as np
import time as t
import scipy.signal as signal
import scipy.fftpack as fft

def preprocess_signal(audio, fs):
    """Фильтрация и нормализация"""
    b, a = signal.butter(4, 900 / (fs / 2), btype='low')
    audio = signal.filtfilt(b, a, audio)
    audio = audio / np.max(np.abs(audio))
    return audio

def cepstral_analysis(audio, fs):
    """Спектральный анализ через цепстр"""
    spectrum = np.log(np.abs(fft.fft(audio)) + 1e-10)
    cepstrum = fft.ifft(spectrum)
    return np.abs(cepstrum[:len(cepstrum) // 2])

def autocorrelation_method(audio, fs, min_freq=50, max_freq=500):
    """Автокорреляция для определения основного тона"""
    autocorr = np.correlate(audio, audio, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    min_lag = fs // max_freq
    max_lag = fs // min_freq
    autocorr[:min_lag] = 0  
    peak_index = np.argmax(autocorr[min_lag:max_lag]) + min_lag
    pitch = fs / peak_index if peak_index > 0 else 0
    return pitch

def yaapt_realtime(audio, fs):
    """Оптимизированный YAAPT для реального времени"""
    audio = preprocess_signal(audio, fs)
    cepstrum = cepstral_analysis(audio, fs)
    pitch = autocorrelation_method(audio, fs)
    return pitch

def process_audio(audio_file, blocksize=1024, plot_results=True):
    """Обработка аудио с использованием YAAPT"""
    times = []
    frequencies = []
    processing_times = []
    
    with StreamFromFile(filename=audio_file, blocksize=blocksize) as stream:
        print(f"Частота дискретизации: {stream.samplerate} Гц")
        stream.set_methods(UnpackRawInFloat32())
        current_time = 0
        
        for i in range(stream.get_iterations()):
            try:
                raw_data = stream.read(blocksize)
                if not raw_data:
                    print(f"Блок {i} пустой, пропускаем...")
                    continue
                start_time = t.time()
                signal = stream.chain_of_methods(raw_data)
                pitch = yaapt_realtime(signal, stream.samplerate)
                times.append(current_time)
                frequencies.append(pitch)
                current_time += blocksize / stream.samplerate
                end_time = t.time()
                processing_times.append(end_time - start_time)
            except Exception as err:
                print(f"Ошибка при обработке блока {i}: {err}")
    
    if processing_times:
        avg_processing_time = sum(processing_times) / len(processing_times)
        print(f"Среднее время обработки одного блока ({blocksize} сэмплов): {avg_processing_time:.6f} секунд")
    else:
        print("Обработка не производилась, данные отсутствуют.")
    
    if plot_results and times and frequencies:
        plt.figure(figsize=(12, 8))
        plt.plot(times, frequencies, label="Основной тон (YAAPT)", color="blue")
        plt.xlabel("Время (с)")
        plt.ylabel("Частота (Гц)")
        plt.title("График основного тона YAAPT")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("Данные для построения графика отсутствуют.")
    
    return times, frequencies

audio1 = 'silero_vad/files/femalechist2.wav' 
process_audio(audio1, blocksize=1024, plot_results=True);