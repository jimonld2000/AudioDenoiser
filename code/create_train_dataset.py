import os
import random
import librosa
import numpy as np
import soundfile as sf

# Pedalboard imports for Reverb
from pedalboard import Pedalboard, Reverb

# ------------------------------------------------------------------------
# PATHS/CONSTANTS
# ------------------------------------------------------------------------
CLEAN_AUDIO_DIR = "./data/train/clean"
NOISE_AUDIO_DIR = "./data/train/noise"
OUTPUT_BASEDIR = "./data/train_processed"    # subfolders: white/, urban/, reverb/, noise_cancellation/
DEBUG_AUDIO_DIR = "./data/debug_audio"       # separate folder for debug .wav

os.makedirs(OUTPUT_BASEDIR, exist_ok=True)
os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)

SAMPLE_RATE = 8000
CHUNK_SECONDS = 2.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)  # 2s * 8000 = 16000

# STFT parameters
N_FFT = 512
HOP_LENGTH = 128

# For debugging: we will save up to this many .wav for each noise type
NUM_DEBUG_WAV = 2

# Desired SNR in dB for “white” and “urban”
SNR_DB = 5.0

# Our noise types
NOISE_TYPES = ["white", "urban", "reverb", "noise_cancellation"]


# ------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------

def load_wav_list(dirname):
    """Return a list of .wav filepaths in given dirname."""
    all_files = []
    for f in os.listdir(dirname):
        if f.lower().endswith(".wav"):
            all_files.append(os.path.join(dirname, f))
    return sorted(all_files)


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


def frame_audio(audio, chunk_samples=CHUNK_SAMPLES, hop_samples=None):
    """
    Frame the audio into chunks of size chunk_samples, no overlap by default.
    Return a list of frames (each length chunk_samples).
    """
    if hop_samples is None:
        hop_samples = chunk_samples

    frames = []
    i = 0
    while i + chunk_samples <= len(audio):
        frames.append(audio[i:i + chunk_samples])
        i += hop_samples
    return frames


def pedalboard_reverb(clean_audio):
    """
    Apply Pedalboard Reverb to a mono audio numpy array.
    Modify room_size/damping/wet_level as you like.
    """
    # Create a pedalboard with a single Reverb effect
    board = Pedalboard([
        Reverb(room_size=0.9, damping=0.9, wet_level=0.33)
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


def audio_to_magnitude_spectrogram(audio_1d):
    """
    Given 1D audio, compute linear magnitude spectrogram using STFT,
    then return the 2D array (freq_bins x time_frames).
    """
    stft_complex = librosa.stft(
        audio_1d,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=False  # avoid padding
    )
    mag, _ = librosa.magphase(stft_complex)
    return mag  # shape ~ (257, 63) for 2s chunk, depends on hop/N_FFT


# ------------------------------------------------------------------------
# MAIN CREATION
# ------------------------------------------------------------------------

def create_train_dataset(clean_dir, noise_dir, output_base, chunk_seconds=2.0):
    """
    1) Load clean audio files, segment into 2s chunks.
    2) For each chunk, apply each noise type => produce a 1D noisy array.
    3) Convert both clean & noisy 1D audio into STFT magnitude spectrograms (2D).
    4) Save them as .npy files (each starting with "clean_" or "noisy_").
    5) Write up to 2 debug WAVs per noise type in a separate debug folder.
    """
    # 1) Gather clean and noise files
    clean_files = load_wav_list(clean_dir)
    noise_files = load_wav_list(noise_dir)

    # 2) Make subdirs for each noise type
    for nt in NOISE_TYPES:
        nt_subdir = os.path.join(output_base, nt)
        os.makedirs(nt_subdir, exist_ok=True)

    # Keep counters for debug WAV and naming
    debug_counts = {nt: 0 for nt in NOISE_TYPES}
    chunk_counter = 0

    # 3) Process each clean file
    for cf in clean_files:
        y_clean, _ = librosa.load(cf, sr=SAMPLE_RATE)

        # Break audio into 2s frames, no overlap
        frames = frame_audio(
            audio=y_clean,
            chunk_samples=CHUNK_SAMPLES,
            hop_samples=CHUNK_SAMPLES
        )

        for frame in frames:
            # Potentially pick random noise for "urban"
            nf = random.choice(noise_files) if len(noise_files) > 0 else None
            if nf and os.path.isfile(nf):
                y_noise, _ = librosa.load(nf, sr=SAMPLE_RATE)
            else:
                y_noise = np.array([], dtype=np.float32)

            # For each noise type
            for nt in NOISE_TYPES:
                
                print(f"Processing noiste type: {nt}\n")
                # 1D noisy chunk
                noisy_chunk = add_noise(
                    clean_audio=frame,
                    noise_audio=y_noise,
                    noise_type=nt,
                    snr_db=SNR_DB
                )

                # Debug WAV: save up to NUM_DEBUG_WAV
                if debug_counts[nt] < NUM_DEBUG_WAV:
                    debug_wav_path = os.path.join(
                        DEBUG_AUDIO_DIR, f"debug_{nt}_{debug_counts[nt]}.wav"
                    )
                    sf.write(debug_wav_path, noisy_chunk, SAMPLE_RATE)
                    debug_counts[nt] += 1

                # --- Convert both clean and noisy 1D to 2D spectrogram
                clean_mag = audio_to_magnitude_spectrogram(frame)
                noisy_mag = audio_to_magnitude_spectrogram(noisy_chunk)

                # Save them individually as .npy
                # Name must start with "clean" or "noisy" for your data_loader
                nt_subdir = os.path.join(output_base, nt)
                noisy_filename = f"noisy_{nt}_chunk_{chunk_counter}.npy"
                clean_filename = f"clean_{nt}_chunk_{chunk_counter}.npy"

                np.save(os.path.join(nt_subdir, noisy_filename),
                        noisy_mag.astype(np.float32))
                np.save(os.path.join(nt_subdir, clean_filename),
                        clean_mag.astype(np.float32))

            chunk_counter += 1

    print("Done! Saved 2D spectrograms in .npy for each noise type.")
    print(f"Debug WAVs are in: {DEBUG_AUDIO_DIR}")
    print("Check subdirectories in:", output_base, "for your spectrogram files.")


if __name__ == "__main__":
    create_train_dataset(
        clean_dir=CLEAN_AUDIO_DIR,
        noise_dir=NOISE_AUDIO_DIR,
        output_base=OUTPUT_BASEDIR,
        chunk_seconds=CHUNK_SECONDS
    )
    print("Done creating train dataset.")
