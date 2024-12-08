import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import convolve

# Define paths
train_path = "./data/train"
noise_path = os.path.join(train_path, "noise")
noisy_path = os.path.join(train_path, "noisy")

#   Path to your Impulse Response (IR) file

ir_path = "./data/reverb/example_reverb.wav"

# Load the impulse response (IR) for reverb
ir, sr_ir = librosa.load(ir_path, sr=16000)

# Normalize the IR
ir = ir / np.max(np.abs(ir))

# Function to apply reverb using convolution


def apply_reverb(audio, ir):
    """
    Apply reverb to an audio signal using convolution with an impulse response.
    """
    reverb_audio = convolve(audio, ir, mode="same")
    # Normalize the output
    reverb_audio = reverb_audio / np.max(np.abs(reverb_audio))
    return reverb_audio


# Filter files for reverb effect
reverb_noise_files = [f for f in os.listdir(noise_path) if "reverb" in f]
reverb_noisy_files = [f for f in os.listdir(noisy_path) if "reverb" in f]

# Reprocess files with realistic reverb
for noise_file, noisy_file in zip(reverb_noise_files, reverb_noisy_files):
    noise_path_full = os.path.join(noise_path, noise_file)
    noisy_path_full = os.path.join(noisy_path, noisy_file)

    print(f"Processing noise file: {noise_path_full}")
    print(f"Processing noisy file: {noisy_path_full}")

    # Load noise and noisy audio
    noise_audio, sr = librosa.load(noise_path_full, sr=16000)
    noisy_audio, _ = librosa.load(noisy_path_full, sr=16000)

    # Apply reverb
    noise_reverb = apply_reverb(noise_audio, ir)
    noisy_reverb = apply_reverb(noisy_audio, ir)

    # Save the updated files
    sf.write(noise_path_full, noise_reverb, sr)
    sf.write(noisy_path_full, noisy_reverb, sr)

print("Reverb processing complete.")
