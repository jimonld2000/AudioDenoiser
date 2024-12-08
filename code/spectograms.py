import librosa
import os
import numpy as np

# Paths to audio directories
clean_audio_dir = './data/train_processed/clean'
noisy_audio_dir = './data/train_processed/noisy'
output_clean_dir = './data/preprocessed/clean'
output_noisy_dir = './data/preprocessed/noisy'

# Ensure output directories exist
os.makedirs(output_clean_dir, exist_ok=True)
os.makedirs(output_noisy_dir, exist_ok=True)

# Parameters
sample_rate = 16000
n_fft = 1024
hop_length = 512
max_length = 200  # Fixed number of frames


def preprocess_and_save(audio_dir, output_dir, max_length, is_noisy=False):
    """
    Preprocess audio files into spectrograms and save as .npy files.

    Args:
        audio_dir (str): Path to input audio files.
        output_dir (str): Path to save spectrograms.
        max_length (int): Maximum number of time frames.
        is_noisy (bool): Whether the audio files are noisy versions.
    """
    audio_files = sorted(
        [f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    for file in audio_files:
        file_path = os.path.join(audio_dir, file)
        audio, _ = librosa.load(file_path, sr=16000)
        spec = librosa.stft(audio, n_fft=1024, hop_length=512)
        spec = np.abs(spec) / np.max(np.abs(spec))  # Normalize magnitudes

        # Pad or truncate to fixed size
        target_height = 512  # Ensure divisible by powers of 2
        if spec.shape[0] > target_height:
            spec = spec[:target_height, :]
        elif spec.shape[0] < target_height:
            pad_height = target_height - spec.shape[0]
            spec = np.pad(spec, ((0, pad_height), (0, 0)), mode='constant')

        if spec.shape[1] > max_length:
            spec = spec[:, :max_length]
        elif spec.shape[1] < max_length:
            pad_width = max_length - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')

        # Save spectrogram as .npy
        base_name = os.path.splitext(file)[0]
        if is_noisy:
            # Append noisy version ID
            base_name += f"_{file.split('_')[-1].replace('.wav', '')}"
        np.save(os.path.join(output_dir, base_name + '.npy'), spec)
        print(f"Saved: {os.path.join(output_dir, base_name + '.npy')}")


# Preprocess clean and noisy audio
preprocess_and_save(clean_audio_dir, output_clean_dir, max_length)
preprocess_and_save(noisy_audio_dir, output_noisy_dir,
                    max_length, is_noisy=True)
