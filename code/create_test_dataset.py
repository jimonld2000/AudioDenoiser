import os
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import convolve
import random

# Define paths
test_path = "./data/test"
clean_path = os.path.join(test_path, "clean")
noise_path = "./data/urban_noise"
noisy_path = os.path.join(test_path, "noisy")
ir_path = "./data/reverb/example_reverb.wav"

# Ensure directories exist
os.makedirs(clean_path, exist_ok=True)
os.makedirs(noisy_path, exist_ok=True)

# Parameters
sample_rate = 16000
target_duration = 3  # seconds
target_length = sample_rate * target_duration  # in samples

# Load the impulse response (IR) for reverb
ir, sr_ir = librosa.load(ir_path, sr=16000)
ir = ir / np.max(np.abs(ir))  # Normalize the IR


def preprocess_audio(audio, target_length):
    """
    Crop or pad audio to a fixed target length.
    """
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        pad_length = target_length - len(audio)
        return np.pad(audio, (0, pad_length), mode="constant")
    return audio


def add_noise(audio, noise_type="white", snr_db=10, urban_noise_path=None):
    """
    Add noise to the audio signal.
    """
    if noise_type == "white":
        noise = np.random.normal(0, 0.3, len(audio))  # White noise

    elif noise_type == "urban" and urban_noise_path:
        noise, _ = librosa.load(urban_noise_path, sr=sample_rate)
        noise = preprocess_audio(noise, len(audio))

    elif noise_type == "reverb":
        noise = convolve(audio, ir, mode="same")
        noise = noise / np.max(np.abs(noise))  # Normalize

    elif noise_type == "noise_cancellation":
        noise = np.zeros_like(audio)
        for i in range(0, len(audio), 16000):
            if random.random() < 0.2:
                end = min(i + 8000, len(audio))
                noise[i:end] = -0.5 * audio[i:end]

    else:
        noise = np.zeros_like(audio)  # Default to silence for invalid types

    # Calculate scaling factor for SNR
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(audio_power / (10**(snr_db / 10) * noise_power))
    noise = noise * scaling_factor

    return audio + noise, noise


def apply_reverb(audio, ir):
    """
    Apply reverb to an audio signal using convolution with an impulse response.
    """
    reverb_audio = convolve(audio, ir, mode="same")
    # Normalize the output
    reverb_audio = reverb_audio / np.max(np.abs(reverb_audio))
    return reverb_audio


def generate_noisy_versions(clean_audio, clean_file, noise_files, noisy_dir):
    """
    Generate 4 noisy versions of a clean audio file:
    1. Clean + White Noise
    2. Clean + Urban Noise
    3. Clean + Reverb
    4. Clean + Reverb + White Noise
    """
    base_name = os.path.splitext(clean_file)[0]

    # Clean + White Noise
    noisy_audio_1, _ = add_noise(clean_audio, noise_type="white")
    noisy_path_1 = os.path.join(noisy_dir, f"{base_name}_white.wav")
    sf.write(noisy_path_1, noisy_audio_1, sample_rate)
    print(f"Generated: {noisy_path_1}")

    # Clean + Urban Noise
    if noise_files:
        urban_noise_path = random.choice(noise_files)
        noisy_audio_2, _ = add_noise(
            clean_audio, noise_type="urban", urban_noise_path=urban_noise_path)
        noisy_path_2 = os.path.join(noisy_dir, f"{base_name}_urban.wav")
        sf.write(noisy_path_2, noisy_audio_2, sample_rate)
        print(f"Generated: {noisy_path_2}")
    else:
        print("No urban noise files found, skipping urban noise generation.")

    # Clean + Reverb
    noisy_audio_3 = apply_reverb(clean_audio, ir)
    noisy_path_3 = os.path.join(noisy_dir, f"{base_name}_reverb.wav")
    sf.write(noisy_path_3, noisy_audio_3, sample_rate)
    print(f"Generated: {noisy_path_3}")

    # Clean + Reverb + White Noise
    reverb_audio = apply_reverb(clean_audio, ir)
    noisy_audio_4, _ = add_noise(reverb_audio, noise_type="white")
    noisy_path_4 = os.path.join(noisy_dir, f"{base_name}_reverb_white.wav")
    sf.write(noisy_path_4, noisy_audio_4, sample_rate)
    print(f"Generated: {noisy_path_4}")


def main():
    # Gather clean audio files
    clean_files = sorted(
        [f for f in os.listdir(clean_path) if f.endswith(".wav")])

    # Gather noise files
    noise_files = sorted([os.path.join(noise_path, f)
                         for f in os.listdir(noise_path) if f.endswith(".wav")])

    # Generate noisy versions for each clean file
    for clean_file in clean_files:
        clean_path_full = os.path.join(clean_path, clean_file)
        clean_audio, _ = librosa.load(clean_path_full, sr=sample_rate)

        # Preprocess clean audio to fixed length
        clean_audio = preprocess_audio(clean_audio, target_length)

        # Generate noisy versions
        generate_noisy_versions(clean_audio, clean_file,
                                noise_files, noisy_path)


if __name__ == "__main__":
    main()
