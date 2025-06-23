# code/test.py (Final Robust Version for 44.1kHz)

import os
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pesq import pesq
from stoi import stoi

from model import UNet

# --- Configuration ---
MODEL_SAVE_DIR = "./saved_models"
TEST_DATA_DIR = "./data/test_processed"
OUTPUT_DIR = "./data/test_output_44_1kHz_robust"
SAMPLE_RATE = 44100
N_FFT = 510
HOP_LENGTH = 1400
METRIC_SAMPLE_RATE = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def has_nan(tensor):
    """Checks if a torch tensor contains NaN values."""
    return torch.isnan(tensor).any()

# --- All calculation functions remain the same as the previous version ---
# (Utility, SNR, SI-SDR, PESQ, STOI, etc.)
def griffin_lim_reconstruction(spectrogram, n_fft, hop_length, iterations=50, length=None):
    return librosa.griffinlim(spectrogram, n_iter=iterations, hop_length=hop_length, win_length=n_fft, length=length)

def calculate_si_sdr(reference, estimate):
    # Add a check for zero-energy estimate to prevent division by zero.
    if np.sum(estimate**2) == 0:
        return -np.inf  # Return negative infinity if the estimate is silent

    min_len = min(len(reference), len(estimate))
    reference, estimate = reference[:min_len], estimate[:min_len]

    alpha = np.dot(estimate, reference) / np.sum(estimate**2)
    target = alpha * estimate
    noise = reference - target
    
    power_target = np.sum(target**2)
    power_noise = np.sum(noise**2)
    
    return 10 * np.log10(power_target / power_noise) if power_noise > 0 else float('inf')

# ... [Other metric functions like calculate_pesq, calculate_stoi, calculate_snr are unchanged] ...
def calculate_pesq(clean_signal, denoised_signal, original_sr, metric_sr):
    try:
        clean_resampled = librosa.resample(y=clean_signal, orig_sr=original_sr, target_sr=metric_sr)
        denoised_resampled = librosa.resample(y=denoised_signal, orig_sr=original_sr, target_sr=metric_sr)
        return pesq(metric_sr, clean_resampled, denoised_resampled, 'wb')
    except Exception as e:
        # This will now catch the NaN errors more gracefully.
        # print(f"Could not calculate PESQ: {e}")
        return None

def calculate_stoi(clean_signal, denoised_signal, original_sr, metric_sr):
    try:
        clean_resampled = librosa.resample(y=clean_signal, orig_sr=original_sr, target_sr=metric_sr)
        denoised_resampled = librosa.resample(y=denoised_signal, orig_sr=original_sr, target_sr=metric_sr)
        return stoi(clean_resampled, denoised_resampled, metric_sr, extended=False)
    except Exception as e:
        # print(f"Could not calculate STOI: {e}")
        return None
        
def calculate_snr(clean_signal, noisy_signal):
    min_len = min(len(clean_signal), len(noisy_signal))
    clean_signal, noisy_signal = clean_signal[:min_len], noisy_signal[:min_len]
    noise = noisy_signal - clean_signal
    rms_signal = np.sqrt(np.mean(clean_signal**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    return 20 * np.log10(rms_signal / rms_noise) if rms_noise > 0 else float('inf')


# --- Main Testing Logic with Robustness Checks ---

def test_single_noise_type(model, noise_type):
    print(f"\n=== Testing model on noise type: {noise_type} ===")
    
    # --- STEP 1: Check model weights for NaN right after loading ---
    for name, param in model.named_parameters():
        if has_nan(param):
            print(f"!!!!!! CRITICAL ERROR: Model for '{noise_type}' has NaN weights in layer: {name}. !!!!!!")
            print("!!!!!! This model is broken. You MUST retrain with stabilization techniques. !!!!!!")
            return # Stop testing this broken model

    noise_output_dir = os.path.join(OUTPUT_DIR, noise_type)
    os.makedirs(noise_output_dir, exist_ok=True)

    noisy_spectrograms = np.load(os.path.join(TEST_DATA_DIR, f"noisy_{noise_type}.npy"))
    clean_spectrograms = np.load(os.path.join(TEST_DATA_DIR, f"clean_{noise_type}.npy"))
    num_samples = len(noisy_spectrograms)
    print(f"Found {num_samples} test samples for '{noise_type}'")

    original_audio_dir = os.path.join(TEST_DATA_DIR, noise_type)
    noisy_torch = torch.tensor(noisy_spectrograms, dtype=torch.float32).unsqueeze(1)

    # --- STEP 2: Check model output for NaN ---
    with torch.no_grad():
        denoised_torch = model(noisy_torch)
    
    if has_nan(denoised_torch):
        print(f"!!!!!! CRITICAL ERROR: Model for '{noise_type}' produced NaN in its output spectrogram. !!!!!!")
        return

    denoised_spectrograms = denoised_torch.squeeze(1).cpu().numpy()
    input_snr_list, output_sisdr_list, pesq_list, stoi_list = [], [], [], []

    for i in range(num_samples):
        # ... (Visualization code remains the same) ...

        clean_audio, _ = librosa.load(os.path.join(original_audio_dir, f"clean_{i}.wav"), sr=SAMPLE_RATE)
        denoised_spec = denoised_spectrograms[i]

        # --- STEP 3: Check for NaN before audio reconstruction ---
        if np.isnan(denoised_spec).any():
            print(f"Warning: Sample {i} spectrogram from model contains NaN. Skipping.")
            continue
        
        target_length = len(clean_audio)
        denoised_audio = griffin_lim_reconstruction(np.maximum(0, denoised_spec), N_FFT, HOP_LENGTH, length=target_length)

        # --- STEP 4: Check for NaN in final audio before metrics ---
        if np.isnan(denoised_audio).any():
            print(f"Warning: Reconstructed audio for sample {i} contains NaN. Skipping metrics.")
            continue

        sf.write(os.path.join(noise_output_dir, f"denoised_sample_{i}.wav"), denoised_audio, SAMPLE_RATE)
        
        output_sisdr_list.append(calculate_si_sdr(clean_audio, denoised_audio))
        pesq_list.append(calculate_pesq(clean_audio, denoised_audio, SAMPLE_RATE, METRIC_SAMPLE_RATE))
        stoi_list.append(calculate_stoi(clean_audio, denoised_audio, SAMPLE_RATE, METRIC_SAMPLE_RATE))

    # ... (Averaging and reporting code is the same) ...
    pesq_list_valid = [s for s in pesq_list if s is not None]
    stoi_list_valid = [s for s in stoi_list if s is not None]
    avg_output_sisdr = np.mean(output_sisdr_list) if output_sisdr_list else "N/A"
    avg_pesq = np.mean(pesq_list_valid) if pesq_list_valid else "N/A"
    avg_stoi = np.mean(stoi_list_valid) if stoi_list_valid else "N/A"
    print(f"\nAverage Metrics for noise type '{noise_type}':")
    # ...


# The main() function remains the same.
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