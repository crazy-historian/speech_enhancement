import os
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
import math
import soundfile as sf
import librosa
from tabulate import tabulate

# =======================
# Noise Classes (unchanged)
# =======================
from abc import ABC, abstractmethod

class Noise(ABC):
    @abstractmethod
    def apply(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Method to add noise to an audio signal."""
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
    """Sequentially applies a list of noises to an audio signal."""
    noisy_audio = audio.copy()
    for noise in noises:
        noisy_audio = noise.apply(noisy_audio, sampling_rate)
    return noisy_audio

class BackgroundNoise(Noise):
    """
    Class to add background noise loaded from a file. If the noise file is longer than the audio,
    it is trimmed; if it is shorter, it is repeated.
    """
    def __init__(self, noise_file: str, gain: float = 1.0):
        self.noise_file = noise_file
        self.gain = gain
        self.noise, self.noise_sr = sf.read(noise_file)
        if self.noise.ndim > 1:
            self.noise = np.mean(self.noise, axis=1)
    
    def apply(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        if sampling_rate != self.noise_sr:
            noise_resampled = librosa.resample(self.noise, orig_sr=self.noise_sr, target_sr=sampling_rate)
        else:
            noise_resampled = self.noise
        target_len = len(audio)
        if len(noise_resampled) >= target_len:
            noise_used = noise_resampled[:target_len]
        else:
            noise_used = np.resize(noise_resampled, target_len)
        return audio + self.gain * noise_used

def load_background_noises(base_folder: str) -> dict:
    """
    Walks through a base folder and creates a list of BackgroundNoise objects for each subfolder.
    The dictionary keys are formatted as "background_<folder_name>".
    """
    background_scenarios = {}
    for folder in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder)
        if os.path.isdir(folder_path):
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
            if files:
                noise_objects = [BackgroundNoise(noise_file=file) for file in files]
                background_scenarios[f"background_{folder}"] = noise_objects
    return background_scenarios

def process_audio_files(wav_files: list, output_dir: str, background_noise_files: list):
    os.makedirs(output_dir, exist_ok=True)

    # Базовые шумы
    noises = [
        ("gaussian", GaussianNoise(std=0.01)),
        ("uniform", UniformNoise(low=-0.01, high=0.01))
    ]

    # Добавляем фоновые шумы по списку файлов
    for idx, file_path in enumerate(background_noise_files):
        if os.path.isfile(file_path) and file_path.lower().endswith(".wav"):
            try:
                noise = BackgroundNoise(noise_file=file_path)
                noise_name = f"background_{os.path.splitext(os.path.basename(file_path))[0]}"
                noises.append((noise_name, noise))
            except Exception as e:
                print(f"Ошибка при загрузке фонового шума {file_path}: {e}")

    # Применяем шумы к каждому аудиофайлу
    for file_path in wav_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        audio, sr = librosa.load(file_path, sr=None)

        for noise_name, noise in noises:
            noisy_audio = noise.apply(audio, sr)
            out_file = os.path.join(output_dir, f"{filename}_{noise_name}.wav")
            sf.write(out_file, noisy_audio, sr)
            print(f"Saved: {out_file}")




if __name__ == "__main__":
    input_files = [
        r"E:\mai\Diploma\speech_enhancement\VAD_testing\Vino2.wav",
        r"E:\mai\Diploma\speech_enhancement\VAD_testing\vinograd2.wav",
        r"E:\mai\Speech\speech_enhancement\audios\ребенок_4_5.wav",
        r"E:\mai\Speech\speech_enhancement\audios\учитель_6.wav",
        r"E:\mai\Speech\speech_enhancement\audios\ребенок_4.wav",
        ] 
    
    background_noise_paths = [r"E:\mai\Diploma\speech_enhancement\DPipe\data\background_noise\knocking\knocking.wav",
                              r"E:\mai\Diploma\speech_enhancement\DPipe\data\background_noise\breath\breath.wav"] 
    
    output_directory = r"E:\mai\Diploma\speech_enhancement\VAD_testing\noisy_files"

    process_audio_files(input_files, output_directory, background_noise_paths)
