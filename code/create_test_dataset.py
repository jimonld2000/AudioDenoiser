# code/create_test_dataset.py (Final Corrected Version)

import os
import glob
import librosa
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, Reverb
from pydub import AudioSegment

# --- Configuration ---
TEST_DATA_DIR_RAW = "./data/test/clean"
TEST_DATA_DIR_PROCESSED = "./data/test_processed"
URBAN_SOUND_DIR = "./data/test/noise"
SAMPLE_RATE = 8000
AUDIO_SECONDS = 2 # Adjusted to match your files (16000 samples / 8000 Hz)
N_FFT = 255
HOP_LENGTH_FFT = 63
SNR_DB = 8.0

os.makedirs(TEST_DATA_DIR_PROCESSED, exist_ok=True)

# --- Utility Functions ---

def audio_to_spectrogram(audio, n_fft, hop_length):
    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    return np.abs(spectrogram)

def add_noise(clean_audio, noise, snr_db):
    if len(noise) < len(clean_audio):
        ratio = int(np.ceil(len(clean_audio) / len(noise)))
        noise = np.tile(noise, ratio)
    noise = noise[:len(clean_audio)]
    
    clean_rms = np.sqrt(np.mean(clean_audio**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    if noise_rms == 0: return clean_audio
    
    snr = 10**(snr_db / 20)
    scale = clean_rms / (noise_rms * snr)
    noisy_audio = clean_audio + noise * scale
    return noisy_audio

def pedalboard_reverb(audio, sample_rate):
    board = Pedalboard([Reverb(room_size=0.9, damping=0.9, wet_level=0.33)])
    return board(audio, sample_rate)

def noise_cancellation_effect(audio):
    """
    Applies a low-pass filter and ensures the output is float32.
    """
    # pydub works with integers, so we need to convert from float
    # Assuming input 'audio' is float32 in [-1.0, 1.0]
    int_audio = (audio * 32767).astype(np.int16)

    segment = AudioSegment(
        int_audio.tobytes(), frame_rate=SAMPLE_RATE,
        sample_width=int_audio.dtype.itemsize, channels=1
    )
    low_passed = segment.low_pass_filter(2000)
    
    # --- FIX ---
    # Convert back to a numpy array of integers
    processed_int_array = np.array(low_passed.get_array_of_samples())
    # Convert back to float32 and normalize to [-1.0, 1.0]
    processed_float_array = processed_int_array.astype(np.float32) / 32768.0
    return processed_float_array

def process_test_audio(clean_file_path, noise_type, noise_samples):
    try:
        clean_audio, _ = librosa.load(clean_file_path, sr=SAMPLE_RATE, duration=AUDIO_SECONDS, res_type='kaiser_fast')

        required_length = AUDIO_SECONDS * SAMPLE_RATE
        if len(clean_audio) < required_length:
            print(f"    -> Skipping {os.path.basename(clean_file_path)}: Audio is shorter than {required_length} samples.")
            return None, None

        if noise_type == "white":
            noisy_audio = add_noise(clean_audio, np.random.randn(len(clean_audio)), SNR_DB)
        elif noise_type == "urban":
            if not noise_samples: return None, None
            noise_sample = noise_samples[np.random.randint(len(noise_samples))]
            noisy_audio = add_noise(clean_audio, noise_sample, SNR_DB)
        elif noise_type == "reverb":
            noisy_audio = pedalboard_reverb(clean_audio, SAMPLE_RATE)
        elif noise_type == "noise_cancellation":
            noisy_audio = noise_cancellation_effect(clean_audio)
        else:
            return None, None

        clean_spectrogram = audio_to_spectrogram(clean_audio, N_FFT, HOP_LENGTH_FFT)
        noisy_spectrogram = audio_to_spectrogram(noisy_audio, N_FFT, HOP_LENGTH_FFT)
        
        return (clean_audio, noisy_audio), (clean_spectrogram, noisy_spectrogram)
    
    except Exception as e:
        print(f"    -> Error processing {clean_file_path}: {e}")
        return None, None

def main():
    print("Starting test dataset creation...")
    clean_files = glob.glob(os.path.join(TEST_DATA_DIR_RAW, "*.wav"))
    print(f"Found {len(clean_files)} clean audio files in '{TEST_DATA_DIR_RAW}'.")
    if not clean_files:
        print("Error: No .wav files found.")
        return

    urban_files = glob.glob(os.path.join(URBAN_SOUND_DIR, "*.wav"))
    print(f"Found {len(urban_files)} noise audio files in '{URBAN_SOUND_DIR}'.")
    
    urban_samples = [librosa.load(f, sr=SAMPLE_RATE, duration=AUDIO_SECONDS)[0] for f in urban_files]
    if not urban_samples:
        print("Warning: No urban noise samples were loaded.")

    noise_types = ["white", "urban", "reverb", "noise_cancellation"]
    
    for noise in noise_types:
        print(f"--- Processing noise type: {noise} ---")
        all_clean_specs, all_noisy_specs = [], []
        
        wav_output_dir = os.path.join(TEST_DATA_DIR_PROCESSED, noise)
        os.makedirs(wav_output_dir, exist_ok=True)
        
        sample_count = 0
        for i, file_path in enumerate(clean_files):
            audio_data, spec_data = process_test_audio(file_path, noise, urban_samples)
            if audio_data and spec_data:
                clean_audio, noisy_audio = audio_data
                clean_spec, noisy_spec = spec_data
                
                sf.write(os.path.join(wav_output_dir, f"clean_{sample_count}.wav"), clean_audio, SAMPLE_RATE)
                sf.write(os.path.join(wav_output_dir, f"noisy_{sample_count}.wav"), noisy_audio, SAMPLE_RATE)

                all_clean_specs.append(clean_spec)
                all_noisy_specs.append(noisy_spec)
                sample_count += 1

        if all_clean_specs:
            np.save(os.path.join(TEST_DATA_DIR_PROCESSED, f"clean_{noise}.npy"), np.array(all_clean_specs))
            np.save(os.path.join(TEST_DATA_DIR_PROCESSED, f"noisy_{noise}.npy"), np.array(all_noisy_specs))
            print(f"----> Successfully saved {len(all_clean_specs)} samples for '{noise}' noise.")

    print("\nTest dataset creation complete.")

if __name__ == '__main__':
    main()