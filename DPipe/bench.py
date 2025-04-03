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

# =======================
# Audio Metrics (unchanged)
# =======================
class AudioMetrics:
    def __init__(self):
        pass

    def compute_snr(self, clean: np.ndarray, processed: np.ndarray) -> float:
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean((clean - processed) ** 2)
        if noise_power < 1e-10:
            return float('inf')
        snr = 10 * math.log10(signal_power / noise_power)
        return snr

    def compute_segmental_snr(self, clean: np.ndarray, processed: np.ndarray, frame_size: int = 400, overlap: int = 200) -> float:
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
        clean_spec = np.abs(np.fft.rfft(clean, n=n_fft)) + 1e-8
        processed_spec = np.abs(np.fft.rfft(processed, n=n_fft)) + 1e-8
        log_clean = np.log10(clean_spec)
        log_processed = np.log10(processed_spec)
        lsd = np.sqrt(np.mean((log_clean - log_processed) ** 2))
        return lsd
    
    def compute_mse(self, clean: np.ndarray, processed: np.ndarray) -> float:
        mse = np.mean((clean - processed) ** 2)
        return mse

    def compute_mae(self, clean: np.ndarray, processed: np.ndarray) -> float:
        mae = np.mean(np.abs(clean - processed))
        return mae

    def compute_metrics(self, clean: np.ndarray, processed: np.ndarray) -> dict:
        return {
            "SNR": self.compute_snr(clean, processed),
            "Segmental_SNR": self.compute_segmental_snr(clean, processed),
            "LSD": self.compute_lsd(clean, processed),
            "MSE": self.compute_mse(clean, processed),
            "MAE": self.compute_mae(clean, processed),
            "Processed_sum": sum(processed),
        }

# =======================
# Base DenoiseModel (unchanged)
# =======================
class DenoiseModel(nn.Module):
    def __init__(self):
        super(DenoiseModel, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def improve(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.no_grad():
            denoised_tensor = self.forward(audio_tensor)
        return denoised_tensor.squeeze(0).numpy()

# =======================
# DeepFilterNet Integration
# =======================
class DeepFilterNetDenoiser(DenoiseModel):
    def __init__(self):
        super(DeepFilterNetDenoiser, self).__init__()
        # Import DeepFilterNet helper functions.
        from df.enhance import enhance, init_df
        self.enhance = enhance
        # Initialize DeepFilterNet. This loads the default model.
        self.model, self.df_state, _ = init_df()
        # Expected sampling rate for the model.
        self.expected_sr = self.df_state.sr()

    def improve(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        # Resample if the input sampling rate is different from the expected.
        if sampling_rate != self.expected_sr:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=self.expected_sr)
        import torch
        # Convert the NumPy array to a float tensor.
        audio_tensor = torch.from_numpy(audio).float()
        # Ensure the audio tensor is 2D: [channels, time]. If it's 1D, add a channel dimension.
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Now shape is [1, T]
        # Run enhancement using DeepFilterNet.
        enhanced_tensor = self.enhance(self.model, self.df_state, audio_tensor)
        # Convert the output tensor back to a NumPy array.
        enhanced = enhanced_tensor.cpu().numpy()
        # Optionally, resample back to the original rate if needed.
        if sampling_rate != self.expected_sr:
            # Squeeze extra dimensions before resampling.
            enhanced = librosa.resample(enhanced.squeeze(), orig_sr=self.expected_sr, target_sr=sampling_rate)
        return enhanced



# =======================
# Pipeline Execution
# =======================
def run_pipeline(only_background=False):
    # Load dataset.
    dataset = load_dataset("Nexdata/Russian_Children_Spontaneous_Speech_Data", 
                           cache_dir=r"E:\mai\Diploma\speech_enhancement\DPipe\cache")['train']
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize the DeepFilterNet-based denoiser.
    model = DeepFilterNetDenoiser()
    metrics = AudioMetrics()

    if not only_background:
        gaussian_noise = GaussianNoise(std=0.02)
        uniform_noise = UniformNoise(low=-0.02, high=0.02)
        noise_scenarios = {
            "clean": [],
            "gaussian": [gaussian_noise],
            "uniform": [uniform_noise],
            "combined": [gaussian_noise, uniform_noise]
        }
    else:
        noise_scenarios = {}

    # Load background noises.
    bg_folder = r"E:\mai\Diploma\speech_enhancement\DPipe\data\background_noise"
    bg_scenarios = load_background_noises(bg_folder)
    noise_scenarios.update(bg_scenarios)
    
    baseline_results = {}
    improved_results = {}
    background_results = {}
    
    for scenario, noise_list in noise_scenarios.items():
        print(f"\nScenario: {scenario}")
        baseline_metrics_list = []
        improved_metrics_list = []
        
        for sample in dataset:
            audio_info = sample["audio"]
            audio = audio_info["array"]
            sampling_rate = audio_info["sampling_rate"]
            
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
                metrics_baseline = {key: np.mean([m[key] for m in baseline_for_sample]) 
                                    for key in baseline_for_sample[0].keys()}
                metrics_improved = {key: np.mean([m[key] for m in improved_for_sample]) 
                                    for key in improved_for_sample[0].keys()}
            else:
                noisy_audio = audio.copy()
                for noise in noise_list:
                    noisy_audio = noise.apply(noisy_audio, sampling_rate)
                metrics_baseline = metrics.compute_metrics(audio, noisy_audio)
                denoised_audio = model.improve(noisy_audio, sampling_rate)
                metrics_improved = metrics.compute_metrics(audio, denoised_audio)
            
            baseline_metrics_list.append(metrics_baseline)
            improved_metrics_list.append(metrics_improved)
        
        avg_baseline = {key: np.mean([m[key] for m in baseline_metrics_list]) 
                        for key in baseline_metrics_list[0].keys()}
        avg_improved = {key: np.mean([m[key] for m in improved_metrics_list]) 
                        for key in improved_metrics_list[0].keys()}
        baseline_results[scenario] = avg_baseline
        improved_results[scenario] = avg_improved
        
        table_data = []
        for metric in avg_baseline.keys():
            table_data.append([metric, f"{avg_baseline[metric]:.2f}", f"{avg_improved[metric]:.2f}"])
        print("Metrics:")
        headers = ["Metric", "Baseline", "Denoised"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        if scenario.startswith("background_"):
            background_results[scenario] = (avg_baseline, avg_improved)
    
    if background_results:
        final_table = []
        headers = ["Folder", "SNR (base)", "SNR (denoised)",
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
        print("\nFinal table for background noises:")
        print(tabulate(final_table, headers=headers, tablefmt="grid"))
        
if __name__ == '__main__':
    run_pipeline(only_background=False)
