import torch
from torch.utils.data import Dataset
import os
import numpy as np


class PreprocessedAudioDenoisingDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, max_length=200):
        """
        Dataset for loading preprocessed spectrograms for audio denoising.

        Args:
            clean_dir (str): Directory containing clean spectrogram files.
            noisy_dir (str): Directory containing noisy spectrogram files.
            max_length (int): Maximum number of time frames for spectrograms.
        """
        self.clean_files = sorted(
            [f for f in os.listdir(clean_dir) if f.endswith('.npy')])
        self.noisy_files = sorted(
            [f for f in os.listdir(noisy_dir) if f.endswith('.npy')])
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.max_length = max_length

        # Map clean files to corresponding noisy files
        self.file_mapping = self._build_file_mapping()

    def _build_file_mapping(self):
        """
        Map each clean file to its corresponding noisy files.
        Assumes that noisy files are named in the format:
        - clean_audio_01.npy -> noisy_audio_01_1.npy, noisy_audio_01_2.npy, etc.
        """
        file_mapping = {}
        for clean_file in self.clean_files:
            base_name = clean_file.split('.')[0]
            noisy_versions = [f for f in self.noisy_files if base_name in f]
            file_mapping[clean_file] = noisy_versions
        return file_mapping

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load clean spectrogram
        clean_file = self.clean_files[idx]
        clean_path = os.path.join(self.clean_dir, clean_file)
        clean_spec = np.load(clean_path)

        # Randomly select one noisy version
        noisy_files = self.file_mapping[clean_file]
        noisy_file = np.random.choice(noisy_files)
        noisy_path = os.path.join(self.noisy_dir, noisy_file)
        noisy_spec = np.load(noisy_path)

        # Convert to tensors
        clean_spec = torch.tensor(clean_spec, dtype=torch.float32).unsqueeze(
            0)  # Add channel dimension
        noisy_spec = torch.tensor(noisy_spec, dtype=torch.float32).unsqueeze(
            0)  # Add channel dimension

        return clean_spec, noisy_spec
