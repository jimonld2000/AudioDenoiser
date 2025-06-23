import os
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pesq

from model import UNet
from loss import CombinedPerceptualLoss

# --------------------------------------------------
# Paths & Parameters
# --------------------------------------------------
TEST_DATA_DIR = "./data/test_processed"     # Contains clean_{noise}.npy, noisy_{noise}.npy
SAVED_MODELS_DIR = "./saved_models"           # Contains unet_denoiser_{noise}.pth
OUTPUT_DIR = "./data/test_output_ensemble"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 8000
N_FFT = 512
HOP_LENGTH_FFT = 128

# Match the noise types for training
NOISE_TYPES = ["white", "urban", "reverb", "noise_cancellation"]

# --------------------------------------------------
# Metric Calculation Functions                  
# --------------------------------------------------
def calculate_snr(clean_signal, noisy_signal):
    """Calculates the Signal-to-Noise Ratio (SNR) in dB."""
    # Ensure signals are numpy arrays
    clean_signal = np.asarray(clean_signal)
    noisy_signal = np.asarray(noisy_signal)
    
    # The noise is the difference between the noisy and clean signals
    noise = noisy_signal - clean_signal
    
    # Calculate RMS of signal and noise
    rms_signal = np.sqrt(np.mean(clean_signal**2))
    rms_noise = np.sqrt(np.mean(noise**2))
    
    # Avoid division by zero
    if rms_noise == 0:
        return float('inf')
        
    snr = 20 * np.log10(rms_signal / rms_noise)
    return snr

def calculate_pesq(clean_signal, denoised_signal, sample_rate=8000):
    """Calculates the Perceptual Evaluation of Speech Quality (PESQ)."""
    if sample_rate not in [8000, 16000]:
        raise ValueError("PESQ is only supported for 8kHz or 16kHz sample rates.")
    try:
        return pesq(sample_rate, clean_signal, denoised_signal, 'nb') # 'nb' for narrowband
    except Exception as e:
        print(f"Could not calculate PESQ: {e}")
        return None


# --------------------------------------------------
# Griffin-Lim Reconstruction 
# --------------------------------------------------
def griffin_lim_reconstruction(magnitude_spectrogram, n_fft, hop_length, iterations=50):
    """
    Reconstruct audio from linear magnitude spectrogram using Griffin-Lim.
    magnitude_spectrogram: shape (freq_bins, time_frames).
    Returns 1D audio array.
    """
    # Start with random phase
    angles = np.exp(2j * np.pi * np.random.rand(*magnitude_spectrogram.shape))
    complex_spec = magnitude_spectrogram * angles

    for _ in range(iterations):
        audio = librosa.istft(complex_spec, hop_length=hop_length) # Inverse STFT
        new_complex_spec = librosa.stft( # STFT
            audio, n_fft=n_fft, hop_length=hop_length
        )
        magnitude = np.abs(new_complex_spec)
        phase = np.angle(new_complex_spec)
        complex_spec = magnitude * np.exp(1j * phase) # Estimated complex spectrogram

    return librosa.istft(complex_spec, hop_length=hop_length)


# --------------------------------------------------
# Utility: Load Model
# --------------------------------------------------
def load_model_for_noise(noise_type):
    """
    Loads the specialized UNet for the given noise_type,
    e.g. unet_denoiser_white.pth, etc.
    """
    model_path = os.path.join(SAVED_MODELS_DIR, f"unet_denoiser_{noise_type}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = UNet(in_channels=1, num_classes=1)
    # Weights only is recommended for loading on CPU
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    print(f"Loaded model for noise type '{noise_type}' from: {model_path}")
    return model


# --------------------------------------------------
# Testing Function
# --------------------------------------------------

def test_single_noise_type(model, noise_type, test_data_dir, output_dir):
    """
    Tests a single noise-type model, calculates objective metrics (SNR, PESQ),
    and saves all relevant outputs.
    """
    print(f"\n=== Testing model on noise type: {noise_type} ===")

    clean_path = os.path.join(test_data_dir, f"clean_{noise_type}.npy")
    noisy_path = os.path.join(test_data_dir, f"noisy_{noise_type}.npy")

    if not (os.path.exists(clean_path) and os.path.exists(noisy_path)):
        print(f"Skipping {noise_type}, missing data files.")
        return

    clean_spectrograms = np.load(clean_path)
    noisy_spectrograms = np.load(noisy_path)
    num_samples = len(noisy_spectrograms)
    print(f"Found {num_samples} test samples for '{noise_type}'")

    # --- Run model to get denoised spectrograms ---
    noisy_torch = torch.tensor(noisy_spectrograms, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        denoised_torch = model(noisy_torch)
        denoised_spectrograms = denoised_torch.squeeze(1).cpu().numpy()

    # --- Initialize lists to store metrics ---
    input_snr_list, output_snr_list, pesq_list = [], [], []

    # --- Process each sample to get audio and calculate metrics ---
    for i in range(num_samples):
        # Reconstruct audio from spectrograms using Griffin-Lim
        clean_audio = griffin_lim_reconstruction(clean_spectrograms[i], N_FFT, HOP_LENGTH_FFT)
        noisy_audio = griffin_lim_reconstruction(noisy_spectrograms[i], N_FFT, HOP_LENGTH_FFT)
        denoised_audio = griffin_lim_reconstruction(denoised_spectrograms[i], N_FFT, HOP_LENGTH_FFT)

        # Save some example audio files
        if i < 5:
            sf.write(os.path.join(output_dir, f"{noise_type}_clean_sample_{i}.wav"), clean_audio, SAMPLE_RATE)
            sf.write(os.path.join(output_dir, f"{noise_type}_noisy_sample_{i}.wav"), noisy_audio, SAMPLE_RATE)
            sf.write(os.path.join(output_dir, f"{noise_type}_denoised_sample_{i}.wav"), denoised_audio, SAMPLE_RATE)

        # --- Calculate metrics ---
        input_snr = calculate_snr(clean_audio, noisy_audio)
        output_snr = calculate_snr(clean_audio, denoised_audio)
        pesq_score = calculate_pesq(clean_audio, denoised_audio, SAMPLE_RATE)
        
        input_snr_list.append(input_snr)
        output_snr_list.append(output_snr)
        if pesq_score is not None:
            pesq_list.append(pesq_score)

    # --- Calculate and print average metrics ---
    avg_input_snr = np.mean(input_snr_list)
    avg_output_snr = np.mean(output_snr_list)
    avg_pesq = np.mean(pesq_list) if pesq_list else "N/A"

    print(f"\nAverage Metrics for noise type '{noise_type}':")
    print(f"  - Average Input SNR: {avg_input_snr:.2f} dB")
    print(f"  - Average Output SNR: {avg_output_snr:.2f} dB")
    print(f"  - Average PESQ: {avg_pesq if isinstance(avg_pesq, str) else f'{avg_pesq:.2f}'}")

    # --- Save metrics to a text file ---
    metrics_file_path = os.path.join(output_dir, f"{noise_type}_metrics.txt")
    with open(metrics_file_path, "w") as f:
        f.write(f"Objective metrics for noise type '{noise_type}':\n")
        f.write(f"Average Input SNR: {avg_input_snr:.2f} dB\n")
        f.write(f"Average Output SNR: {avg_output_snr:.2f} dB\n")
        f.write(f"Average PESQ: {avg_pesq if isinstance(avg_pesq, str) else f'{avg_pesq:.2f}'}\n")

    # --- Plot spectrogram comparisons (no changes needed here) ---
    for i in range(min(5, num_samples)):
        plt.figure(figsize=(12, 6))
        def plot_spectrogram(spec, title, pos):
            plt.subplot(1, 3, pos)
            plt.title(title)
            plt.imshow(spec, aspect='auto', origin='lower', cmap='magma')
            plt.colorbar(format='%+2.0f dB')
        
        plot_spectrogram(noisy_spectrograms[i], "Noisy Spectrogram", 1)
        plot_spectrogram(denoised_spectrograms[i], "Denoised Spectrogram", 2)
        plot_spectrogram(clean_spectrograms[i], "Clean Spectrogram", 3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{noise_type}_spectrogram_{i}.png"))
        plt.close()
# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    print("Starting specialized test for each noise type...")

    for noise_type in NOISE_TYPES:
        # 1) Load the specialized model (if available)
        try:
            model = load_model_for_noise(noise_type)
        except FileNotFoundError:
            print(f"Model for noise type '{noise_type}' not found. Skipping.")
            print(f"Model for noise type '{noise_type}' not found. Skipping.")
            continue

        # 2) Test on corresponding test data
        test_single_noise_type(
            model=model,
            noise_type=noise_type,
            test_data_dir=TEST_DATA_DIR,
            output_dir=OUTPUT_DIR
        )
