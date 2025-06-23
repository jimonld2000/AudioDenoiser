# code/test.py (44.1kHz Version with Resampling for Metrics)

import os
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi

from model import UNet

# --- Configuration for 44.1kHz ---
MODEL_SAVE_DIR = "./saved_models"
TEST_DATA_DIR = "./data/test_processed"
OUTPUT_DIR = "./data/test_output_44_1kHz"
SAMPLE_RATE = 44100  # CHANGED

# --- FFT Parameters must match the data creation script ---
N_FFT = 510          # CHANGED
HOP_LENGTH = 1400    # CHANGED

# --- Metrics Configuration ---
# PESQ and STOI require 8kHz or 16kHz. We'll resample to 16kHz for wideband calculation.
METRIC_SAMPLE_RATE = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Utility Functions ---

def griffin_lim_reconstruction(spectrogram, n_fft, hop_length, iterations=50, length=None):
    return librosa.griffinlim(spectrogram, n_iter=iterations, hop_length=hop_length, win_length=n_fft, length=length)

# --- Metric Calculation Functions (Now with Resampling) ---

def calculate_snr(clean_signal, noisy_signal):
    # SNR is calculated at the original sample rate.
    min_len = min(len(clean_signal), len(noisy_signal))
    clean_signal, noisy_signal = clean_signal[:min_len], noisy_signal[:min_len]
    noise = noisy_signal - clean_signal
    rms_signal = np.sqrt(np.mean(clean_signal**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    return 20 * np.log10(rms_signal / rms_noise) if rms_noise > 0 else float('inf')

def calculate_si_sdr(reference, estimate):
    # SI-SDR is also calculated at the original sample rate.
    min_len = min(len(reference), len(estimate))
    reference, estimate = reference[:min_len], estimate[:min_len]
    if np.sum(reference**2) == 0: return -np.inf
    alpha = np.dot(estimate, reference) / np.sum(estimate**2)
    target, noise = alpha * estimate, reference - (alpha * estimate)
    power_target, power_noise = np.sum(target**2), np.sum(noise**2)
    return 10 * np.log10(power_target / power_noise) if power_noise > 0 else float('inf')

def calculate_pesq(clean_signal, denoised_signal, original_sr, metric_sr):
    """Resamples audio to metric_sr before calculating PESQ."""
    try:
        clean_resampled = librosa.resample(y=clean_signal, orig_sr=original_sr, target_sr=metric_sr)
        denoised_resampled = librosa.resample(y=denoised_signal, orig_sr=original_sr, target_sr=metric_sr)
        return pesq(metric_sr, clean_resampled, denoised_resampled, 'wb')  # Use 'wb' for wideband
    except Exception as e:
        print(f"Could not calculate PESQ: {e}")
        return None

def calculate_stoi(clean_signal, denoised_signal, original_sr, metric_sr):
    """Resamples audio to metric_sr before calculating STOI."""
    try:
        clean_resampled = librosa.resample(y=clean_signal, orig_sr=original_sr, target_sr=metric_sr)
        denoised_resampled = librosa.resample(y=denoised_signal, orig_sr=original_sr, target_sr=metric_sr)
        return stoi(clean_resampled, denoised_resampled, metric_sr, extended=False)
    except Exception as e:
        print(f"Could not calculate STOI: {e}")
        return None

# --- Main Testing Logic with Visualization ---

def test_single_noise_type(model, noise_type):
    print(f"\n=== Testing model on noise type: {noise_type} ===")
    
    noise_output_dir = os.path.join(OUTPUT_DIR, noise_type)
    os.makedirs(noise_output_dir, exist_ok=True)

    noisy_spectrograms = np.load(os.path.join(TEST_DATA_DIR, f"noisy_{noise_type}.npy"))
    clean_spectrograms = np.load(os.path.join(TEST_DATA_DIR, f"clean_{noise_type}.npy"))
    num_samples = len(noisy_spectrograms)
    print(f"Found {num_samples} test samples for '{noise_type}'")

    original_audio_dir = os.path.join(TEST_DATA_DIR, noise_type)

    noisy_torch = torch.tensor(noisy_spectrograms, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        denoised_torch = model(noisy_torch)
    denoised_spectrograms = denoised_torch.squeeze(1).cpu().numpy()

    input_snr_list, output_sisdr_list, pesq_list, stoi_list = [], [], [], []

    for i in range(num_samples):
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        librosa.display.specshow(librosa.amplitude_to_db(noisy_spectrograms[i], ref=np.max), ax=axes[0], sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear')
        axes[0].set_title('Noisy Spectrogram (Input)')
        librosa.display.specshow(librosa.amplitude_to_db(denoised_spectrograms[i], ref=np.max), ax=axes[1], sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear')
        axes[1].set_title('Denoised Spectrogram (Model Output)')
        librosa.display.specshow(librosa.amplitude_to_db(clean_spectrograms[i], ref=np.max), ax=axes[2], sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear')
        axes[2].set_title('Clean Spectrogram (Ground Truth)')
        plt.tight_layout()
        plt.savefig(os.path.join(noise_output_dir, f"spectrogram_comparison_{i}.png"))
        plt.close(fig)

        # Audio Reconstruction
        clean_audio, _ = librosa.load(os.path.join(original_audio_dir, f"clean_{i}.wav"), sr=SAMPLE_RATE)
        noisy_audio, _ = librosa.load(os.path.join(original_audio_dir, f"noisy_{i}.wav"), sr=SAMPLE_RATE)
        target_length = len(clean_audio)
        denoised_spec_for_recon = np.maximum(0, denoised_spectrograms[i])
        denoised_audio = griffin_lim_reconstruction(denoised_spec_for_recon, N_FFT, HOP_LENGTH, length=target_length)
        sf.write(os.path.join(noise_output_dir, f"denoised_sample_{i}.wav"), denoised_audio, SAMPLE_RATE)

        # Metric Calculation
        input_snr_list.append(calculate_snr(clean_audio, noisy_audio))
        output_sisdr_list.append(calculate_si_sdr(clean_audio, denoised_audio))
        pesq_list.append(calculate_pesq(clean_audio, denoised_audio, SAMPLE_RATE, METRIC_SAMPLE_RATE))
        stoi_list.append(calculate_stoi(clean_audio, denoised_audio, SAMPLE_RATE, METRIC_SAMPLE_RATE))

    # Averaging and Reporting
    pesq_list_valid = [s for s in pesq_list if s is not None]
    stoi_list_valid = [s for s in stoi_list if s is not None]
    avg_input_snr = np.mean(input_snr_list)
    avg_output_sisdr = np.mean(output_sisdr_list)
    avg_pesq = np.mean(pesq_list_valid) if pesq_list_valid else "N/A"
    avg_stoi = np.mean(stoi_list_valid) if stoi_list_valid else "N/A"

    print(f"\nAverage Metrics for noise type '{noise_type}':")
    print(f"  - Average Input SNR: {avg_input_snr:.2f} dB")
    print(f"  - Average Output SI-SDR: {avg_output_sisdr:.2f} dB")
    print(f"  - Average PESQ (wb): {avg_pesq if isinstance(avg_pesq, str) else f'{avg_pesq:.3f}'}")
    print(f"  - Average STOI: {avg_stoi if isinstance(avg_stoi, str) else f'{avg_stoi:.3f}'}")


def main():
    print("Starting specialized test for each noise type at 44.1kHz...")
    noise_types = ["white", "urban", "reverb", "noise_cancellation"]
    
    for noise_type in noise_types:
        model_path = os.path.join(MODEL_SAVE_DIR, f"unet_denoiser_{noise_type}.pth")
        if not os.path.exists(model_path):
            print(f"Model for '{noise_type}' not found.")
            continue

        model = UNet(in_channels=1, num_classes=1)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        test_single_noise_type(model, noise_type)

if __name__ == '__main__':
    main()