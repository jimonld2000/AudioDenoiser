# code/test.py (Final Version)

import os
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pesq import pesq # Using the more robust 'pesq' library

from model import UNet

# --- Configuration ---
MODEL_SAVE_DIR = "./saved_models"
TEST_DATA_DIR = "./data/test_processed"
OUTPUT_DIR = "./data/test_output_ensemble"
SAMPLE_RATE = 8000
N_FFT = 254
HOP_LENGTH_FFT = 63

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Utility Functions ---

def griffin_lim_reconstruction(spectrogram, n_fft, hop_length, iterations=50, length=None):
    """Reconstructs audio from a magnitude spectrogram using the Griffin-Lim algorithm."""
    return librosa.griffinlim(spectrogram, n_iter=iterations, hop_length=hop_length, win_length=n_fft, length=length)

# --- Metric Calculation Functions ---

def calculate_snr(clean_signal, noisy_signal):
    """Calculates the Signal-to-Noise Ratio (SNR) in dB."""
    min_len = min(len(clean_signal), len(noisy_signal))
    clean_signal = clean_signal[:min_len]
    noisy_signal = noisy_signal[:min_len]
    
    noise = noisy_signal - clean_signal
    
    rms_signal = np.sqrt(np.mean(clean_signal**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    
    if rms_noise == 0:
        return float('inf')
        
    snr = 20 * np.log10(rms_signal / rms_noise)
    return snr

def calculate_pesq(clean_signal, denoised_signal, sample_rate=8000):
    """Calculates PESQ using the robust 'pesq' library."""
    if sample_rate not in [8000, 16000]:
        print(f"Warning: PESQ is only supported for 8kHz or 16kHz. Skipping for SR={sample_rate}.")
        return None
    try:
        # The 'pesq' library works directly with float audio arrays.
        min_len = min(len(clean_signal), len(denoised_signal))
        clean_s = clean_signal[:min_len]
        denoised_s = denoised_signal[:min_len]

        # Check for silent signals which can cause errors
        if np.sum(np.abs(clean_s)) == 0 or np.sum(np.abs(denoised_s)) == 0:
            print("Could not calculate PESQ: A signal is silent.")
            return None
            
        return pesq(sample_rate, clean_s, denoised_s, 'nb') # 'nb' for narrowband
    except Exception as e:
        print(f"Could not calculate PESQ: {e}")
        return None

# --- Main Testing Logic ---

def test_single_noise_type(model, noise_type):
    """
    Tests a single noise-type model, calculates objective metrics (SNR, PESQ),
    and saves all relevant outputs.
    """
    print(f"\n=== Testing model on noise type: {noise_type} ===")
    
    noisy_spectrogram_path = os.path.join(TEST_DATA_DIR, f"noisy_{noise_type}.npy")
    if not os.path.exists(noisy_spectrogram_path):
        print(f"Skipping {noise_type}, missing data file.")
        return

    noisy_spectrograms = np.load(noisy_spectrogram_path)
    num_samples = len(noisy_spectrograms)
    print(f"Found {num_samples} test samples for '{noise_type}'")

    original_audio_dir = os.path.join(TEST_DATA_DIR, noise_type)

    noisy_torch = torch.tensor(noisy_spectrograms, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        denoised_torch = model(noisy_torch)
    denoised_spectrograms = denoised_torch.squeeze(1).cpu().numpy()

    input_snr_list, output_snr_list, pesq_list = [], [], []

    for i in range(num_samples):
        try:
            clean_audio, _ = librosa.load(os.path.join(original_audio_dir, f"clean_{i}.wav"), sr=SAMPLE_RATE)
            noisy_audio, _ = librosa.load(os.path.join(original_audio_dir, f"noisy_{i}.wav"), sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Warning: Could not load original audio for sample {i}. Skipping. Error: {e}")
            continue

        target_length = len(clean_audio)
        denoised_audio = griffin_lim_reconstruction(denoised_spectrograms[i], N_FFT, HOP_LENGTH_FFT, length=target_length)
        sf.write(os.path.join(OUTPUT_DIR, f"{noise_type}_denoised_sample_{i}.wav"), denoised_audio, SAMPLE_RATE)

        input_snr_list.append(calculate_snr(clean_audio, noisy_audio))
        output_snr_list.append(calculate_snr(clean_audio, denoised_audio))
        
        pesq_score = calculate_pesq(clean_audio, denoised_audio, SAMPLE_RATE)
        if pesq_score is not None: pesq_list.append(pesq_score)

    avg_input_snr = np.mean(input_snr_list) if input_snr_list else "N/A"
    avg_output_snr = np.mean(output_snr_list) if output_snr_list else "N/A"
    avg_pesq = np.mean(pesq_list) if pesq_list else "N/A"

    print(f"\nAverage Metrics for noise type '{noise_type}':")
    print(f"  - Average Input SNR: {avg_input_snr if isinstance(avg_input_snr, str) else f'{avg_input_snr:.2f} dB'}")
    print(f"  - Average Output SNR: {avg_output_snr if isinstance(avg_output_snr, str) else f'{avg_output_snr:.2f} dB'}")
    print(f"  - Average PESQ: {avg_pesq if isinstance(avg_pesq, str) else f'{avg_pesq:.2f}'}")

def main():
    print("Starting specialized test for each noise type...")
    noise_types = ["white", "urban", "reverb", "noise_cancellation"]
    
    for noise_type in noise_types:
        model_path = os.path.join(MODEL_SAVE_DIR, f"unet_denoiser_{noise_type}.pth")
        if not os.path.exists(model_path):
            print(f"Model for '{noise_type}' not found.")
            continue

        print(f"Loaded model for noise type '{noise_type}' from: {model_path}")
        model = UNet(in_channels=1)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        test_single_noise_type(model, noise_type)

if __name__ == '__main__':
    main()
