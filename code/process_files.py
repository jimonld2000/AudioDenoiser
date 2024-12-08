import os
import librosa
import soundfile as sf
import numpy as np

# Paths to dataset directories
clean_dir = './data/train/clean'
noisy_dir = './data/train/noisy'
noise_dir = './data/train/noise'

# Output directories (optional: to save processed files separately)
processed_clean_dir = './data/train_processed/clean'
processed_noisy_dir = './data/train_processed/noisy'
processed_noise_dir = './data/train_processed/noise'

# Ensure output directories exist
os.makedirs(processed_clean_dir, exist_ok=True)
os.makedirs(processed_noisy_dir, exist_ok=True)
os.makedirs(processed_noise_dir, exist_ok=True)

# Parameters
sample_rate = 16000  # Target sample rate
target_duration = 3  # Target duration in seconds
target_length = int(sample_rate * target_duration)  # Target length in samples


def preprocess_audio(file_path, output_path, target_length, sample_rate):
    """
    Preprocess audio by cropping or padding to a fixed length.

    Args:
        file_path (str): Path to the input audio file.
        output_path (str): Path to save the processed audio file.
        target_length (int): Target length in samples.
        sample_rate (int): Target sample rate.
    """
    audio, _ = librosa.load(file_path, sr=sample_rate)

    # Crop or pad audio to target length
    if len(audio) > target_length:  # Crop
        audio = audio[:target_length]
    elif len(audio) < target_length:  # Pad
        pad_length = target_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')

    # Save processed audio
    sf.write(output_path, audio, sample_rate)


# Preprocess all files in a directory
def preprocess_directory(input_dir, output_dir, target_length, sample_rate):
    """
    Preprocess all audio files in a directory.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
        target_length (int): Target length in samples.
        sample_rate (int): Target sample rate.
    """
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    for file_name in files:
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        preprocess_audio(input_path, output_path, target_length, sample_rate)
        print(f"Processed: {input_path} -> {output_path}")


# Preprocess all directories
print("Processing clean files...")
preprocess_directory(clean_dir, processed_clean_dir,
                     target_length, sample_rate)

print("Processing noisy files...")
preprocess_directory(noisy_dir, processed_noisy_dir,
                     target_length, sample_rate)

print("Processing noise files...")
preprocess_directory(noise_dir, processed_noise_dir,
                     target_length, sample_rate)

print("All files processed successfully!")
