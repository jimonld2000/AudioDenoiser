import os
import shutil
import librosa
import numpy as np
import soundfile as sf
import random
from scipy.signal import convolve

# Define paths
dataset_path = "..\\IRMAS-TrainingData\\IRMAS-TrainingData"
train_path = ".\\data\\train"
clean_path = os.path.join(train_path, "clean")
noise_path = os.path.join(train_path, "noise")
noisy_path = os.path.join(train_path, "noisy")

# Ensure directories exist
os.makedirs(clean_path, exist_ok=True)
os.makedirs(noise_path, exist_ok=True)
os.makedirs(noisy_path, exist_ok=True)

# Step 1: Move clean files to 'clean' folder
instrument_folders = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
for instrument_folder in instrument_folders:
    for file in os.listdir(instrument_folder):
        if file.endswith(".wav"):
            source_path = os.path.join(instrument_folder, file)
            dest_path = os.path.join(clean_path, file)
            shutil.copy(source_path, dest_path)

# Step 2: Generate noisy versions of each clean file
def add_noise(audio, noise_type="white", snr_db=10, urban_noise_path=None):

    if noise_type == "white":
        noise = np.random.normal(0, 0.3, len(audio))  #  white noise

    elif noise_type == "urban" and urban_noise_path:
        noise, _ = librosa.load(urban_noise_path, sr=16000)
        if len(noise) > len(audio):
            noise = noise[:len(audio)]
        else:
            noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))[:len(audio)]

    elif noise_type == "reverb":
        impulse_response = np.zeros(500)
        impulse_response[0] = 1
        impulse_response[100] = 0.5
        noise = convolve(audio, impulse_response, mode='same')

    elif noise_type == "noise_cancellation":
        noise = np.zeros_like(audio)
        for i in range(0, len(audio), 16000):
            if random.random() < 0.2:
                end = min(i + 8000, len(audio))
                noise[i:end] = -0.5 * audio[i:end]

    else:
        noise = np.zeros_like(audio)  

    # Calculate scaling factor for SNR
    audio_power = np.mean(audio ** 2)
    noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(audio_power / (10**(snr_db / 10) * noise_power))
    noise = noise * scaling_factor

    return audio + noise, noise

# Process each clean file
for file in os.listdir(clean_path):
    if file.endswith(".wav"):
        audio_path = os.path.join(clean_path, file)
        audio, sr = librosa.load(audio_path, sr=16000)

        for noise_type in ["white", "urban", "reverb", "noise_cancellation"]:
            if noise_type == "urban":
                urban_noise_files = [f for f in os.listdir(noise_path) if f.endswith(".wav")]
                if urban_noise_files:
                    urban_noise_file = random.choice(urban_noise_files)
                    urban_noise_path = os.path.join(noise_path, urban_noise_file)
                    noisy_audio, noise = add_noise(audio, noise_type=noise_type, urban_noise_path=urban_noise_path)
                else:
                    print(f"No urban noise files found in {noise_path}")
                    continue
            else:
                noisy_audio, noise = add_noise(audio, noise_type=noise_type)

            # Save the noise and noisy files
            noise_filename = f"{file.split('.')[0]}_{noise_type}_noise.wav"
            noisy_filename = f"{file.split('.')[0]}_{noise_type}_noisy.wav"

            sf.write(os.path.join(noise_path, noise_filename), noise, sr)
            sf.write(os.path.join(noisy_path, noisy_filename), noisy_audio, sr)

print("Data preparation complete.")
