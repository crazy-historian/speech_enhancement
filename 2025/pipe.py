import os
import glob
import argparse
import torch
import torchaudio
import torch.nn as nn
import numpy as np

# Optional: Install and import the pesq package (pip install pesq) for PESQ score.
try:
    from pesq import pesq
except ImportError:
    pesq = None

#########################################
# Noise Generation Functions
#########################################
def generate_white_noise(shape):
    """Generate white noise with the given shape."""
    return torch.randn(shape)

def generate_pink_noise(shape):
    """A simple pink noise generator by low-pass filtering white noise."""
    noise = torch.randn(shape)
    alpha = 0.9
    pink = torch.zeros_like(noise)
    pink[:, 0] = noise[:, 0]
    for i in range(1, noise.shape[1]):
        pink[:, i] = alpha * pink[:, i - 1] + (1 - alpha) * noise[:, i]
    return pink

def generate_sine_noise(shape, sr, freq=60):
    """Generate a sine-wave noise (simulating a hum) at a given frequency (Hz)."""
    t = torch.arange(0, shape[1], dtype=torch.float32) / sr
    sine_wave = torch.sin(2 * torch.pi * freq * t)
    # Match the number of channels by unsqueezing and expanding
    return sine_wave.unsqueeze(0).expand(shape[0], -1)

def add_noise_to_signal(clean: torch.Tensor, sr: int, noise_type: str, snr_db: float) -> torch.Tensor:
    """
    Adds noise of a specified type to the clean signal at the desired SNR.
    SNR is defined in dB: snr = 10 * log10(P_signal / P_noise).
    """
    if noise_type == "white":
        noise = generate_white_noise(clean.shape)
    elif noise_type == "pink":
        noise = generate_pink_noise(clean.shape)
    elif noise_type == "sine":
        noise = generate_sine_noise(clean.shape, sr)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Calculate power and adjust noise level
    signal_power = torch.mean(clean ** 2)
    desired_noise_power = signal_power / (10 ** (snr_db / 10))
    current_noise_power = torch.mean(noise ** 2) + 1e-8  # avoid div by zero
    noise_scaling = torch.sqrt(desired_noise_power / current_noise_power)
    noise = noise * noise_scaling
    noisy_signal = clean + noise
    return noisy_signal

#########################################
# Dummy Noise Suppression Model & Wrapper
#########################################
class DummyDenoiser(nn.Module):
    """
    A placeholder noise suppression model.
    Replace this with your actual model as needed.
    """
    def __init__(self):
        super(DummyDenoiser, self).__init__()
        # Example: a simple 1D convolution layer for mono audio
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x shape: (batch, channels, samples)
        return self.conv(x)

class NoiseSuppressionModelWrapper:
    """
    Wraps any PyTorch noise suppression model to provide a consistent interface.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()  # set to evaluation mode

    def process(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """
        Process the noisy audio tensor.
        Expects input shape (batch, channels, samples).
        """
        with torch.no_grad():
            denoised_audio = self.model(noisy_audio)
        return denoised_audio

#########################################
# Metrics Computation
#########################################
def compute_snr(clean: torch.Tensor, processed: torch.Tensor) -> float:
    """
    Computes the Signal-to-Noise Ratio (SNR) in dB.
    clean: ground truth clean signal.
    processed: noisy or denoised signal.
    """
    noise = clean - processed
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2) + 1e-8
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

def compute_pesq_metric(clean: torch.Tensor, processed: torch.Tensor, sr: int) -> float:
    """
    Computes the PESQ score (if the pesq package is available).
    """
    if pesq is None:
        return None
    # Remove batch/channel dimensions (assuming mono) and convert to numpy arrays
    clean_np = clean.squeeze().cpu().numpy()
    processed_np = processed.squeeze().cpu().numpy()
    # 'wb' stands for wideband; adjust if using narrowband audio
    score = pesq(sr, clean_np, processed_np, 'wb')
    return score

#########################################
# Pipeline: Process One Clean Audio File
#########################################
def process_audio_file(clean_file: str, noise_types, snr_levels, wrapper: NoiseSuppressionModelWrapper, output_folder: str):
    # Load the clean audio file
    clean, sr = torchaudio.load(clean_file)  # shape: (channels, samples)
    # Ensure mono audio (if multichannel, average them)
    if clean.shape[0] != 1:
        clean = clean.mean(dim=0, keepdim=True)
    
    metrics_all = {}
    for noise_type in noise_types:
        for snr_db in snr_levels:
            # Create a noisy version from the clean audio
            noisy = add_noise_to_signal(clean, sr, noise_type, snr_db)
            
            # Process the noisy audio through the noise suppression model
            noisy_batch = noisy.unsqueeze(0)  # shape: (1, channels, samples)
            denoised_batch = wrapper.process(noisy_batch)
            denoised = denoised_batch.squeeze(0)
            
            # Compute SNR metrics
            snr_noisy = compute_snr(clean, noisy)
            snr_denoised = compute_snr(clean, denoised)
            snr_improvement = snr_denoised - snr_noisy

            metric = {
                'snr_noisy': snr_noisy,
                'snr_denoised': snr_denoised,
                'snr_improvement': snr_improvement,
            }
            pesq_score = compute_pesq_metric(clean, denoised, sr)
            if pesq_score is not None:
                metric['pesq'] = pesq_score

          
            metric['mae_clean_noisy'] = torch.nn.functional.l1_loss(clean, noisy)
            metric['mae_clean_denoised'] = torch.nn.functional.l1_loss(clean, denoised)
            
            # Use a key that combines noise type and SNR level
            key = f"{noise_type}_snr{snr_db}dB"
            metrics_all[key] = metric
            
            # Save the noisy and denoised audio for inspection
            base_name = os.path.splitext(os.path.basename(clean_file))[0]
            noisy_out_path = os.path.join(output_folder, f"{base_name}_{key}_noisy.wav")
            denoised_out_path = os.path.join(output_folder, f"{base_name}_{key}_denoised.wav")
            torchaudio.save(noisy_out_path, noisy, sr)
            torchaudio.save(denoised_out_path, denoised, sr)
    
    return metrics_all

#########################################
# Main Pipeline
#########################################
def main():
    parser = argparse.ArgumentParser(description="Noise Suppression Evaluation Pipeline")
    parser.add_argument('--clean_folder', type=str, required=True,
                        help='Path to folder with clean audio files (.wav)')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Folder to save generated noisy and denoised audio files')
    parser.add_argument('--noise_types', nargs='+', default=['white', 'pink', 'sine'],
                        help='List of noise types to add (e.g., white, pink, sine)')
    parser.add_argument('--snr_levels', nargs='+', type=float, default=[5, 10, 15],
                        help='List of SNR levels (in dB) at which to add noise')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # Instantiate your noise suppression model (replace DummyDenoiser with your actual model)
    model = DummyDenoiser()
    wrapper = NoiseSuppressionModelWrapper(model)
    
    # Get the list of clean audio files
    audio_files = glob.glob(os.path.join(args.clean_folder, '*.wav'))
    if not audio_files:
        print("No WAV files found in the provided clean folder.")
        return
    
    # Process each clean audio file
    for clean_file in audio_files:
        metrics = process_audio_file(clean_file, args.noise_types, args.snr_levels, wrapper, args.output_folder)
        print(f"Processed {os.path.basename(clean_file)}:")
        for condition, metric in metrics.items():
            print(f"  Condition: {condition}")
            for key, value in metric.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.2f}")
                else:
                    print(f"    {key}: {value}")
        print("-" * 40)

if __name__ == '__main__':
    main()

# e:/mai/Speech/.venv/Scripts/python.exe e:/mai/Speech/2025/pipe.py --clean_folder E:\mai\Speech\2025\clean_audio_samples --output_folder E:\mai\Speech\2025\denoised_audio_samples --noise_types white pink sine --snr_levels 5 10 15