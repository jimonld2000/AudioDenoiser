import os
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

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
    Tests a single noise-type model on the test data for that noise type.
    1) Loads clean_{noise_type}.npy and noisy_{noise_type}.npy
    2) Reconstructs & saves noisy audio via Griffin-Lim
    3) Runs model on noisy spectrograms -> saves denoised audio
    4) Plots spectrogram comparisons
    5) Computes & outputs the MSE between the denoised and clean spectrograms.
    """
    print(f"\n=== Testing model on noise type: {noise_type} ===")

    # Paths
    clean_path = os.path.join(test_data_dir, f"clean_{noise_type}.npy")
    noisy_path = os.path.join(test_data_dir, f"noisy_{noise_type}.npy")

    if not (os.path.exists(clean_path) and os.path.exists(noisy_path)):
        print(f"Skipping {noise_type}, missing {clean_path} or {noisy_path}")
        return

    # Load spectrogram arrays: shape (N, freq_bins, time_frames)
    clean_spectrograms = np.load(clean_path)
    noisy_spectrograms = np.load(noisy_path)
    num_samples = len(noisy_spectrograms)
    print(f"Found {num_samples} test samples for noise type '{noise_type}'")

    # Convert to torch tensor: (N, 1, freq_bins, time_frames)
    noisy_torch = torch.tensor(noisy_spectrograms, dtype=torch.float32).unsqueeze(1)

    # Optional: Reconstruct & save a few "noisy" audios via Griffin-Lim
    for i in range(min(5, num_samples)):
        noisy_spec_2d = noisy_spectrograms[i]  # shape (freq_bins, time_frames)
        noisy_audio = griffin_lim_reconstruction(
            noisy_spec_2d, N_FFT, HOP_LENGTH_FFT
        )
        sf.write(os.path.join(output_dir, f"{noise_type}_noisy_{i}.wav"),
                 noisy_audio, SAMPLE_RATE)

    # Pass through model to get denoised spectrograms
    with torch.no_grad():
        denoised_torch = model(noisy_torch)  # (N, 1, freq_bins, time_frames)
        denoised_spectrograms = denoised_torch.squeeze(1).cpu().numpy()
        # shape => (N, freq_bins, time_frames)

    # Compute perceptual losses between denoised and clean spectrograms
    criterion = CombinedPerceptualLoss()
    with torch.no_grad():
        denoised_torch = torch.tensor(denoised_spectrograms, dtype=torch.float32).unsqueeze(1)
        clean_torch = torch.tensor(clean_spectrograms, dtype=torch.float32).unsqueeze(1)
        total_loss, stft_loss, mel_loss, l1_loss = criterion(denoised_torch, clean_torch)

    # Print all loss metrics
    print(f"\nLoss metrics for noise type '{noise_type}':")
    print(f"Total Loss: {total_loss.item():.6f}")
    print(f"STFT Loss: {stft_loss.item():.6f}")
    print(f"Mel Loss: {mel_loss.item():.6f}")
    print(f"L1 Loss: {l1_loss.item():.6f}")

    # Save all metrics to a text file
    metrics_file_path = os.path.join(output_dir, f"{noise_type}_metrics.txt")
    with open(metrics_file_path, "w") as f:
        f.write(f"Perceptual metrics for noise type '{noise_type}':\n")
        f.write(f"Total Loss: {total_loss.item():.6f}\n")
        f.write(f"STFT Loss: {stft_loss.item():.6f}\n")
        f.write(f"Mel Loss: {mel_loss.item():.6f}\n")
        f.write(f"L1 Loss: {l1_loss.item():.6f}\n")

    # Reconstruct & save denoised audio for a few samples
    for i, denoised_spec in enumerate(denoised_spectrograms):
        if i >= 5:
            break
        denoised_audio = griffin_lim_reconstruction(
            denoised_spec, N_FFT, HOP_LENGTH_FFT
        )
        sf.write(os.path.join(
            output_dir, f"{noise_type}_denoised_{i}.wav"), denoised_audio, SAMPLE_RATE)

    # Plot spectrogram comparisons for a few samples
    for i in range(min(5, num_samples)):
        plt.figure(figsize=(12, 6))

        # Noisy spectrogram
        # Plot spectrograms directly with dB scale and fixed range
        def plot_spectrogram(spec, title, pos):
            plt.subplot(1, 3, pos)
            plt.title(title)
            # Convert to dB with reference level
            #spec_db = 20 * np.log10(np.maximum(spec, 1e-6))
            # Set fixed dB range for visualization
            #vmin, vmax = -100, 0  # Fixed range for darker background
            plt.imshow(spec, aspect='auto', origin='lower', 
                      cmap='magma')
            plt.colorbar(format='%+2.0f dB')

        # Plot all three spectrograms
        plot_spectrogram(noisy_spectrograms[i], "Noisy Spectrogram", 1)
        plot_spectrogram(denoised_spectrograms[i], "Denoised Spectrogram", 2)
        plot_spectrogram(clean_spectrograms[i], "Clean Spectrogram", 3)

        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f"{noise_type}_spectrogram_{i}.png"))
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
