# code/test.py (Revised for Robust Metrics)

import os
import torch
import librosa
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

from model import UNet

# --- Configuration ---
MODEL_SAVE_DIR = "./saved_models"
TEST_DATA_DIR = "./data/test_processed"
OUTPUT_DIR = "./data/test_output_revised"
SAMPLE_RATE = 44100
N_FFT = 254
HOP_LENGTH_FFT = 63

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Utility Functions ---

def griffin_lim_reconstruction(spectrogram, n_fft, hop_length, iterations=50, length=None):
    """Reconstructs audio from a magnitude spectrogram using the Griffin-Lim algorithm."""
    return librosa.griffinlim(spectrogram, n_iter=iterations, hop_length=hop_length, win_length=n_fft, length=length)

# --- Metric Calculation Functions ---

def calculate_snr(clean_signal, noisy_signal):
    """
    Calculates the traditional Signal-to-Noise Ratio (SNR) in dB.
    NOTE: Best used for measuring the input noise level, not for evaluating
    phase-agnostic models due to sensitivity to phase errors.
    """
    min_len = min(len(clean_signal), len(noisy_signal))
    clean_signal = clean_signal[:min_len]
    noisy_signal = noisy_signal[:min_len]
    
    # The noise is the difference between the noisy signal and the clean one
    noise = noisy_signal - clean_signal
    
    rms_signal = np.sqrt(np.mean(clean_signal**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    
    if rms_noise == 0:
        return float('inf')
        
    snr = 20 * np.log10(rms_signal / rms_noise)
    return snr

def calculate_si_sdr(reference, estimate):
    """
    Calculates the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    This is the preferred metric for evaluating audio separation and enhancement.
    """
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]

    if np.sum(reference**2) == 0:
        # Prevent division by zero if the reference signal is silence
        return -np.inf

    # Find the optimal scaling factor
    alpha = np.dot(estimate, reference) / np.sum(estimate**2)
    
    target = alpha * estimate
    noise = reference - target
    
    power_target = np.sum(target**2)
    power_noise = np.sum(noise**2)
    
    if power_noise == 0:
        return float('inf')
        
    si_sdr = 10 * np.log10(power_target / power_noise)
    
    return si_sdr

def calculate_pesq(clean_signal, denoised_signal, sample_rate=8000):
    """Calculates PESQ (Perceptual Evaluation of Speech Quality)."""
    if sample_rate not in [8000, 16000]:
        print(f"Warning: PESQ only supports 8kHz or 16kHz. Skipping for SR={sample_rate}.")
        return None
    try:
        min_len = min(len(clean_signal), len(denoised_signal))
        clean_s = clean_signal[:min_len]
        denoised_s = denoised_signal[:min_len]

        if np.sum(np.abs(clean_s)) == 0 or np.sum(np.abs(denoised_s)) == 0:
            print("Could not calculate PESQ: A signal is silent.")
            return None
            
        return pesq(sample_rate, clean_s, denoised_s, 'nb') # 'nb' for narrowband
    except Exception as e:
        print(f"Could not calculate PESQ: {e}")
        return None

def calculate_stoi(clean_signal, denoised_signal, sample_rate):
    """Calculates STOI (Short-Time Objective Intelligibility)."""
    try:
        min_len = min(len(clean_signal), len(denoised_signal))
        clean_s = clean_signal[:min_len]
        denoised_s = denoised_signal[:min_len]
        return stoi(clean_s, denoised_s, sample_rate, extended=False)
    except Exception as e:
        print(f"Could not calculate STOI: {e}")
        return None


# --- Main Testing Logic ---

def test_single_noise_type(model, noise_type):
    """
    Tests a single noise-type model, calculates objective metrics (SI-SDR, PESQ, STOI),
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

    # Lists to store our metrics
    input_snr_list, output_sisdr_list, pesq_list, stoi_list = [], [], [], []

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

        # --- Calculate all metrics ---
        # 1. Input SNR (baseline)
        input_snr_list.append(calculate_snr(clean_audio, noisy_audio))
        
        # 2. Output SI-SDR (primary performance metric)
        output_sisdr_list.append(calculate_si_sdr(clean_audio, denoised_audio))
        
        # 3. PESQ (perceptual quality)
        pesq_score = calculate_pesq(clean_audio, denoised_audio, SAMPLE_RATE)
        if pesq_score is not None:
            pesq_list.append(pesq_score)

        # 4. STOI (intelligibility)
        stoi_score = calculate_stoi(clean_audio, denoised_audio, SAMPLE_RATE)
        if stoi_score is not None:
            stoi_list.append(stoi_score)

    # --- Averaging and Reporting ---
    avg_input_snr = np.mean(input_snr_list) if input_snr_list else "N/A"
    avg_output_sisdr = np.mean(output_sisdr_list) if output_sisdr_list else "N/A"
    avg_pesq = np.mean(pesq_list) if pesq_list else "N/A"
    avg_stoi = np.mean(stoi_list) if stoi_list else "N/A"

    print(f"\nAverage Metrics for noise type '{noise_type}':")
    print(f"  - Average Input SNR: {avg_input_snr if isinstance(avg_input_snr, str) else f'{avg_input_snr:.2f} dB'}")
    print(f"  - Average Output SI-SDR: {avg_output_sisdr if isinstance(avg_output_sisdr, str) else f'{avg_output_sisdr:.2f} dB'}")
    print(f"  - Average PESQ: {avg_pesq if isinstance(avg_pesq, str) else f'{avg_pesq:.3f}'}")
    print(f"  - Average STOI: {avg_stoi if isinstance(avg_stoi, str) else f'{avg_stoi:.3f}'}")


def main():
    print("Starting specialized test for each noise type...")
    noise_types = ["white", "urban", "reverb", "noise_cancellation"]
    
    for noise_type in noise_types:
        model_path = os.path.join(MODEL_SAVE_DIR, f"unet_denoiser_{noise_type}.pth")
        if not os.path.exists(model_path):
            print(f"Model for '{noise_type}' not found.")
            continue

        print(f"Loading model for noise type '{noise_type}' from: {model_path}")
        model = UNet(in_channels=1, num_classes=1) # Ensure num_classes=1 matches your training
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        test_single_noise_type(model, noise_type)

if __name__ == '__main__':
    main()