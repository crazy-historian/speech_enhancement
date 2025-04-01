
import os
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from abc import ABC, abstractmethod
import math
import soundfile as sf
import librosa
from tabulate import tabulate
# =======================
# Интерфейс для шума
# =======================
class Noise(ABC):
    @abstractmethod
    def apply(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Метод для зашумления аудиосигнала"""
        pass

class GaussianNoise(Noise):
    def __init__(self, std: float = 0.01):
        self.std = std

    def apply(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        noise = np.random.normal(0, self.std, audio.shape)
        return audio + noise

class UniformNoise(Noise):
    def __init__(self, low: float = -0.01, high: float = 0.01):
        self.low = low
        self.high = high

    def apply(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        noise = np.random.uniform(self.low, self.high, audio.shape)
        return audio + noise

def apply_noises(audio: np.ndarray, sampling_rate: int, noises: list) -> np.ndarray:
    """
    Применяет список шумов к аудио.
    Можно применять по отдельности или комбинировать.
    """
    noisy_audio = audio.copy()
    for noise in noises:
        noisy_audio = noise.apply(noisy_audio, sampling_rate)
    return noisy_audio

print("ЧТО ДЕЛАТЬ С ФАЙЛАМИ С ГРОМКИМ ШУМОМ")
class BackgroundNoise(Noise):
    """
    Класс для фонового шума, загружаемого из файла.
    Если длина файла больше длины аудио, то он обрезается до нужного размера,
    если меньше – повторяется (с использованием np.resize).
    """
    def __init__(self, noise_file: str, gain: float = 1.0):
        self.noise_file = noise_file
        self.gain = gain
        self.noise, self.noise_sr = sf.read(noise_file)
        # Если аудио стерео, берем среднее по каналам
        if self.noise.ndim > 1:
            self.noise = np.mean(self.noise, axis=1)
    
    def apply(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        # Если частоты дискретизации не совпадают, ресэмплим
        if sampling_rate != self.noise_sr:
            noise_resampled = librosa.resample(self.noise, orig_sr=self.noise_sr, target_sr=sampling_rate)
        else:
            noise_resampled = self.noise
        target_len = len(audio)
        # Если шум длиннее, обрезаем его
        if len(noise_resampled) >= target_len:
            noise_used = noise_resampled[:target_len]
        else:
            # Если короче – повторяем его
            noise_used = np.resize(noise_resampled, target_len)
        return audio + self.gain * noise_used

def apply_noises(audio: np.ndarray, sampling_rate: int, noises: list) -> np.ndarray:
    """
    Применяет список шумов к аудио последовательно (для стандартных сценариев).
    Для фоновых шумов будет использоваться иной подход (см. ниже).
    """
    noisy_audio = audio.copy()
    for noise in noises:
        noisy_audio = noise.apply(noisy_audio, sampling_rate)
    return noisy_audio

# =======================
# Функция загрузки фоновых шумов
# =======================
def load_background_noises(base_folder: str) -> dict:
    """
    Обходит папку base_folder и для каждой подпапки создает список объектов BackgroundNoise.
    Ключ словаря имеет формат "background_<имя_папки>".
    """
    background_scenarios = {}
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            # Фильтруем аудио файлы по расширению, например .wav
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
            if files:
                noise_objects = [BackgroundNoise(noise_file=file) for file in files]
                background_scenarios[f"background_{folder}"] = noise_objects
    return background_scenarios
# =======================
# Интерфейс для метрик
# =======================
class AudioMetrics:
    def __init__(self):
        pass

    def compute_snr(self, clean: np.ndarray, processed: np.ndarray) -> float:
        """
        Вычисляет классический SNR.
        """
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean((clean - processed) ** 2)
        if noise_power < 1e-10:
            return float('inf')
        snr = 10 * math.log10(signal_power / noise_power)
        return snr

    def compute_segmental_snr(self, clean: np.ndarray, processed: np.ndarray, frame_size: int = 400, overlap: int = 200) -> float:
        """
        Вычисляет сегментный SNR по окнам сигнала.
        frame_size - размер окна,
        overlap - перекрытие окон.
        """
        num_samples = len(clean)
        start = 0
        snr_list = []
        
        while start < num_samples:
            end = start + frame_size
            if end > num_samples:
                frame_clean = clean[start:]
                frame_processed = processed[start:]
            else:
                frame_clean = clean[start:end]
                frame_processed = processed[start:end]
            
            signal_power = np.mean(frame_clean ** 2)
            noise_power = np.mean((frame_clean - frame_processed) ** 2)
            if noise_power > 1e-10:
                frame_snr = 10 * math.log10(signal_power / noise_power)
                snr_list.append(frame_snr)
            start += (frame_size - overlap)
        
        if snr_list:
            return np.mean(snr_list)
        else:
            return float('inf')

    def compute_lsd(self, clean: np.ndarray, processed: np.ndarray, n_fft: int = 512) -> float:
        """
        Вычисляет Log-Spectral Distance (LSD) между чистым и обработанным сигналом.
        """
        clean_spec = np.abs(np.fft.rfft(clean, n=n_fft)) + 1e-8
        processed_spec = np.abs(np.fft.rfft(processed, n=n_fft)) + 1e-8
        
        log_clean = np.log10(clean_spec)
        log_processed = np.log10(processed_spec)
        
        lsd = np.sqrt(np.mean((log_clean - log_processed) ** 2))
        return lsd
    
    def compute_mse(self, clean: np.ndarray, processed: np.ndarray) -> float:
        """
        Вычисляет среднеквадратичную ошибку (MSE) между чистым и обработанным сигналом.
        """
        mse = np.mean((clean - processed) ** 2)
        return mse

    def compute_mae(self, clean: np.ndarray, processed: np.ndarray) -> float:
        """
        Вычисляет среднюю абсолютную ошибку (MAE) между чистым и обработанным сигналом.
        """
        mae = np.mean(np.abs(clean - processed))
        return mae

    def compute_metrics(self, clean: np.ndarray, processed: np.ndarray) -> dict:
        """
        Вычисляет и возвращает набор метрик: SNR, сегментный SNR, LSD, MSE и MAE.
        """
        return {
            "SNR": self.compute_snr(clean, processed),
            "Segmental_SNR": self.compute_segmental_snr(clean, processed),
            "LSD": self.compute_lsd(clean, processed),
            "MSE": self.compute_mse(clean, processed),
            "MAE": self.compute_mae(clean, processed)
        }

# =======================
# Модель удаления шумов
# =======================
class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()
        # Пример: простая заглушка без реальной обработки.
        # Здесь можно разместить слои модели, например, сверточные слои или RNN.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # В реальной модели здесь будет логика денойзинга.
        return x

    def improve(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Унифицированный интерфейс для улучшения (удаления шумов) аудио.
        Принимает numpy-массив и возвращает обработанный сигнал.
        """
        # Преобразуем в тензор
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, length]
        # Пропускаем через модель (здесь это заглушка)
        with torch.no_grad():
            denoised_tensor = self.forward(audio_tensor)
        # Возвращаем обратно в numpy (убираем batch dimension)
        return denoised_tensor.squeeze(0).numpy()

# =======================
# Пайплайн проверки модели
# =======================
def run_pipeline():
    # Загрузка датасета.
    # В данном примере используется датасет с русской детской спонтанной речью.
    dataset = load_dataset("Nexdata/Russian_Children_Spontaneous_Speech_Data", 
                           cache_dir="E:\\mai\\Diploma\\speech_enhancement\\DPipe\\cache")['train']
    print(f"Загружено {len(dataset)} примеров")
    
    # Инициализируем модель и метрики.
    model = DenoiseModel()
    metrics = AudioMetrics()

    # Определяем стандартные шумы.
    gaussian_noise = GaussianNoise(std=0.02)
    uniform_noise = UniformNoise(low=-0.02, high=0.02)
    noise_scenarios = {
        "clean": [],
        "gaussian": [gaussian_noise],
        "uniform": [uniform_noise],
        "combined": [gaussian_noise, uniform_noise]
    }
    
    # Загружаем фоновые шумы из указанной папки.
    bg_folder = r"E:\mai\Diploma\speech_enhancement\DPipe\data\background_noise"
    bg_scenarios = load_background_noises(bg_folder)
    # Добавляем фоновые шумы в общий словарь сценариев.
    noise_scenarios.update(bg_scenarios)
    
    # Словари для сохранения итоговых метрик.
    baseline_results = {}
    improved_results = {}
    
    # Для фоновых сценариев отдельно сохраним результаты для итоговой таблицы.
    background_results = {}
    
    # Проходим по каждому сценарию шума.
    for scenario, noise_list in noise_scenarios.items():
        print(f"\nСценарий: {scenario}")
        baseline_metrics_list = []
        improved_metrics_list = []
        
        for sample in dataset:
            # Извлекаем аудио и sampling_rate из датасета.
            audio_info = sample["audio"]
            audio = audio_info["array"]
            sampling_rate = audio_info["sampling_rate"]
            
            # Если сценарий с фоновой шумовой папкой, обрабатываем каждый файл отдельно и усредняем.
            if scenario.startswith("background_"):
                baseline_for_sample = []
                improved_for_sample = []
                for noise in noise_list:
                    noisy_audio = noise.apply(audio, sampling_rate)
                    m_baseline = metrics.compute_metrics(audio, noisy_audio)
                    denoised_audio = model.improve(noisy_audio, sampling_rate)
                    m_improved = metrics.compute_metrics(audio, denoised_audio)
                    baseline_for_sample.append(m_baseline)
                    improved_for_sample.append(m_improved)
                # Усредняем метрики по всем файлам в папке для данного примера.
                metrics_baseline = {key: np.mean([m[key] for m in baseline_for_sample]) 
                                    for key in baseline_for_sample[0].keys()}
                metrics_improved = {key: np.mean([m[key] for m in improved_for_sample]) 
                                    for key in improved_for_sample[0].keys()}
            else:
                # Для остальных сценариев применяем шум(ы) последовательно.
                noisy_audio = apply_noises(audio, sampling_rate, noise_list)
                metrics_baseline = metrics.compute_metrics(audio, noisy_audio)
                denoised_audio = model.improve(noisy_audio, sampling_rate)
                metrics_improved = metrics.compute_metrics(audio, denoised_audio)
            
            baseline_metrics_list.append(metrics_baseline)
            improved_metrics_list.append(metrics_improved)
        
        # Усредняем метрики по всем примерам.
        avg_baseline = {key: np.mean([m[key] for m in baseline_metrics_list]) 
                        for key in baseline_metrics_list[0].keys()}
        avg_improved = {key: np.mean([m[key] for m in improved_metrics_list]) 
                        for key in improved_metrics_list[0].keys()}
        baseline_results[scenario] = avg_baseline
        improved_results[scenario] = avg_improved
        
        # Выводим таблицу для каждого сценария.
        table_data = []
        for metric in avg_baseline.keys():
            table_data.append([metric, f"{avg_baseline[metric]:.2f}", f"{avg_improved[metric]:.2f}"])
        print("Метрики:")
        headers = ["Метрика", "Baseline", "Denoised"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Если сценарий относится к фоновой шумовой папке, сохраняем для итоговой таблицы.
        if scenario.startswith("background_"):
            background_results[scenario] = (avg_baseline, avg_improved)
    
    # Итоговая таблица для фоновых шумов (разбиение по папкам).
    if background_results:
        final_table = []
        headers = ["Папка", "SNR (base)", "SNR (denoised)",
                   "SegSNR (base)", "SegSNR (denoised)",
                   "LSD (base)", "LSD (denoised)",
                   "MSE (base)", "MSE (denoised)",
                   "MAE (base)", "MAE (denoised)"]
        for scenario, (base, denoised) in background_results.items():
            final_table.append([
                scenario,
                f"{base['SNR']:.2f}", f"{denoised['SNR']:.2f}",
                f"{base['Segmental_SNR']:.2f}", f"{denoised['Segmental_SNR']:.2f}",
                f"{base['LSD']:.2f}", f"{denoised['LSD']:.2f}",
                f"{base['MSE']:.4f}", f"{denoised['MSE']:.4f}",
                f"{base['MAE']:.4f}", f"{denoised['MAE']:.4f}"
            ])
        print("\nИтоговая таблица для фоновых шумов:")
        print(tabulate(final_table, headers=headers, tablefmt="grid"))
        
if __name__ == '__main__':
    run_pipeline()


# бэнч 
#- слова
#- протяжные слоги
#- не речевые

from datasets import load_dataset

#ds = load_dataset("Nexdata/Russian_Children_Spontaneous_Speech_Data", cache_dir="E:\mai\Diploma\speech_enhancement\DPipe\cache")
#print(ds['train']['audio'])