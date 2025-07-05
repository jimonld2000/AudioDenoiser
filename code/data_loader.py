import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from tqdm import tqdm
import random

class WavToSpecDataset(Dataset):
    """
    Dataset that loads paired .wav files at their original sample rate,
    converts them to spectrograms on-the-fly, and can use a random subset.
    This version is corrected to work at 44.1 kHz without truncation.
    """
    def __init__(self, data_dir, sample_rate=44100, n_fft=2048, hop_length=512, subset_fraction=1.0):
        self.pairs = []
        self.sample_rate = sample_rate
        
        # Spectrogram transformation
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0, # Power spectrogram for magnitude
            center=True,
            pad_mode="reflect",
        )

        print(f"Scanning for paired .wav files in: {data_dir} at {sample_rate}Hz")
        noisy_dir = os.path.join(data_dir, "noisy")
        clean_dir = os.path.join(data_dir, "clean")

        if not os.path.isdir(noisy_dir) or not os.path.isdir(clean_dir):
            raise FileNotFoundError(f"Could not find 'clean' and 'noisy' subdirectories in {data_dir}")

        # Create a mapping from clean basenames for quick lookup
        clean_files_map = {
            "_".join(f.split('_clean.wav')[0].split('_')): os.path.join(clean_dir, f)
            for f in os.listdir(clean_dir) if f.endswith("_clean.wav")
        }
        
        # Find pairs by parsing noisy filenames
        for noisy_fn in tqdm(os.listdir(noisy_dir), desc="Finding audio pairs"):
            if not noisy_fn.endswith('.wav'):
                continue
            
            base_name = "_".join(noisy_fn.split('_noisy_')[0].split('_'))
            
            if base_name in clean_files_map:
                clean_path = clean_files_map[base_name]
                noisy_path = os.path.join(noisy_dir, noisy_fn)
                self.pairs.append((noisy_path, clean_path))

        if not self.pairs:
            raise RuntimeError(f"No paired .wav files found in {data_dir}. Check filenames.")
        
        print(f"Found {len(self.pairs)} total paired audio files.")

        # Logic to take a random subset of the data
        if subset_fraction < 1.0:
            num_samples = int(len(self.pairs) * subset_fraction)
            print(f"Taking a random {subset_fraction*100:.1f}% subset of the data ({num_samples} samples)...")
            random.shuffle(self.pairs)
            self.pairs = self.pairs[:num_samples]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]

        # Load waveforms
        noisy_wav, sr_n = torchaudio.load(noisy_path)
        clean_wav, sr_c = torchaudio.load(clean_path)

        # Assert sample rate is correct, no resampling needed
        assert sr_n == self.sample_rate and sr_c == self.sample_rate, \
            f"Sample rate mismatch. Expected {self.sample_rate}, but got {sr_n} and {sr_c}"

        # Convert to mono if stereo
        if noisy_wav.shape[0] > 1: noisy_wav = torch.mean(noisy_wav, dim=0, keepdim=True)
        if clean_wav.shape[0] > 1: clean_wav = torch.mean(clean_wav, dim=0, keepdim=True)

        # Generate spectrograms
        noisy_spec = self.spectrogram(noisy_wav)
        clean_spec = self.spectrogram(clean_wav)
        
        # Return magnitude spectrograms (as the U-Net typically predicts magnitude)
        # Using log-magnitude can also be beneficial for training stability
        return torch.log1p(noisy_spec), torch.log1p(clean_spec)
