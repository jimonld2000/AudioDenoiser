import os
import librosa
import numpy as np
import random
# Pedalboard imports for Reverb
from pedalboard import Pedalboard, Reverb

# ------------------------
# Paths
# ------------------------
CLEAN_AUDIO_DIR = "./data/test/clean"
NOISE_DIR = "./data/test/noise"
# IMPULSE_RESPONSE_PATH = "./data/reverb/example_reverb.wav"
OUTPUT_DIR = "./data/test_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# Parameters
# ------------------------
SAMPLE_RATE = 8000
N_FFT = 512
HOP_LENGTH_FFT = 128

# Example of noise types your code might handle:
NOISE_TYPES = ["white", "urban", "reverb", "noise_cancellation"]

# Desired SNR in dB for “white” and “urban”
SNR_DB = 8.0

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------


def audio_to_spectrogram(audio):
    """
    Convert 1D audio array -> linear magnitude STFT spectrogram.
    """
    stft_out = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH_FFT)
    magnitude, _ = librosa.magphase(stft_out)
    return magnitude  # shape: (freq_bins, time_frames)

def match_audio_length(noise, target_len):
    """
    Return a noise array of exactly `target_len` samples by either
    tiling or taking a random snippet.
    """
    if len(noise) == target_len:
        return noise.copy()
    elif len(noise) < target_len:
        # tile
        repeat_factor = int(np.ceil(target_len / len(noise)))
        big = np.tile(noise, repeat_factor)
        return big[:target_len]
    else:
        # noise is longer, pick random snippet
        start = np.random.randint(0, len(noise) - target_len)
        snippet = noise[start:start + target_len]
        return snippet

def pedalboard_reverb(clean_audio):
    """
    Apply Pedalboard Reverb to a mono audio numpy array.
    Modify room_size/damping/wet_level as you like.
    """
    # Create a pedalboard with a single Reverb effect
    board = Pedalboard([
        Reverb(room_size=0.9, damping=0.9, wet_level=0.35)
    ])

    # Pedalboard expects shape (channels, samples). We'll assume mono audio => shape (1, num_samples)
    mono_chunk = np.expand_dims(clean_audio, axis=0)
    # Apply the board
    effected = board(mono_chunk, SAMPLE_RATE)
    # Shape is still (1, num_samples). Squeeze back to (num_samples,)
    return np.squeeze(effected, axis=0)


def add_noise(clean_audio, noise_audio, noise_type, snr_db=SNR_DB):
    """
    Add the specified noise_type to clean_audio:
      - "white": random white noise at SNR = snr_db
      - "urban": random snippet/tile of loaded noise at SNR = snr_db
      - "reverb": use pedalboard reverb
      - "noise_cancellation": partial cancellation segments
    Returns the resulting noisy audio in [-1,1].
    """
    clean_len = len(clean_audio)

    # -- Reverb using Pedalboard
    if noise_type == "reverb":
        # Apply Pedalboard reverb
        out = pedalboard_reverb(clean_audio)
        # Clip to [-1,1] in case the effect has boosted levels
        noisy = np.clip(out, -1.0, 1.0)

    # -- Noise cancellation
    elif noise_type == "noise_cancellation":
        noise = np.zeros_like(clean_audio)
        i = 0
        while i < clean_len:
            end = min(i + 16000, clean_len)  # process in 2s blocks
            if random.random() < 0.8:  # increased probability
                half_end = i + 8000  # half the block
                half_end = min(half_end, end)
                noise[i:half_end] = -0.8 * clean_audio[i:half_end]  # stronger factor
            i += 16000
        noisy = clean_audio + noise
        noisy = np.clip(noisy, -1.0, 1.0)

    # -- White or Urban
    else:
        if noise_type == "white":
            noise_audio = np.random.randn(clean_len)
        else:  # "urban"
            if noise_audio is None or len(noise_audio) == 0:
                noise_audio = np.zeros(clean_len, dtype=np.float32)
            else:
                noise_audio = match_audio_length(noise_audio, clean_len)

        # SNR scaling
        clean_rms = np.sqrt(np.mean(clean_audio ** 2) + 1e-12)
        noise_rms = np.sqrt(np.mean(noise_audio ** 2) + 1e-12)
        snr_linear = 10.0 ** (snr_db / 20.0)  # e.g., SNR=5dB => ~1.78
        desired_noise_rms = clean_rms / snr_linear
        if noise_rms > 1e-9:
            noise_audio *= (desired_noise_rms / noise_rms)
        else:
            noise_audio = np.zeros_like(clean_audio)
        out = clean_audio + noise_audio
        noisy = np.clip(out, -1.0, 1.0)

    return noisy

def process_test_audio(clean_files, noise_files, noise_type):
    clean_spects, noisy_spects = [], []

    for clean_file in clean_files:
        y_clean, _ = librosa.load(clean_file, sr=SAMPLE_RATE)

        # y_clean is the entire 2-second audio
        noise_file = np.random.choice(
            noise_files) if noise_type == "urban" else None
        noise, _ = librosa.load(
            noise_file, sr=SAMPLE_RATE) if noise_file else (None, None)
        noisy = add_noise(y_clean, noise, noise_type,
                          SNR_DB)  # must be the entire 2s

        # convert each to spectrogram
        clean_spects.append(audio_to_spectrogram(y_clean))
        noisy_spects.append(audio_to_spectrogram(noisy))

    # shape: (N, freq_bins, time_frames)
    return np.array(clean_spects), np.array(noisy_spects)


# ------------------------
# Main Script
# ------------------------
if __name__ == "__main__":

    # Gather all clean files
    clean_files = [
        os.path.join(CLEAN_AUDIO_DIR, f)
        for f in os.listdir(CLEAN_AUDIO_DIR)
        if f.endswith(".wav")
    ]
    # Gather all noise files
    noise_files = [
        os.path.join(NOISE_DIR, f)
        for f in os.listdir(NOISE_DIR)
        if f.endswith(".wav")
    ]

    # Process each noise type
    for noise_type in NOISE_TYPES:
        print(f"Processing noise type: {noise_type}")
        clean_batches, noisy_batches = process_test_audio(
            clean_files, noise_files, noise_type
        )

        # e.g. Save to .npy
        np.save(os.path.join(
            OUTPUT_DIR, f"clean_{noise_type}.npy"), clean_batches)
        np.save(os.path.join(
            OUTPUT_DIR, f"noisy_{noise_type}.npy"), noisy_batches)

    print("Test dataset creation is complete!")
