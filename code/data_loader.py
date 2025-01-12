import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

class SpectrogramDataset(Dataset):
    """
    Dataset for loading spectrogram pairs (clean and noisy) from a single folder.
    Expects .npy files starting with 'clean_' or 'noisy_' in data_dir.
    """
    def __init__(self, data_dir, target_size=(256, 64)):
        self.pairs = []
        self.target_size = target_size

        # Directly look for .npy files in data_dir
        clean_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("clean") and f.endswith(".npy")
        ])
        noisy_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("noisy") and f.endswith(".npy")
        ])
        
        print(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files in {data_dir}")
        assert len(clean_files) == len(noisy_files), f"Mismatch in {data_dir}"

        self.pairs.extend(zip(noisy_files, clean_files))
        print(f"Total pairs loaded: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]

        # Load .npy
        noisy = np.load(noisy_path).astype(np.float16)
        clean = np.load(clean_path).astype(np.float16)

        # pad/truncate
        noisy = self._pad_or_truncate(noisy, self.target_size)
        clean = self._pad_or_truncate(clean, self.target_size)

        # Convert to torch
        noisy = torch.tensor(noisy).unsqueeze(0).float()
        clean = torch.tensor(clean).unsqueeze(0).float()

        return noisy, clean

    def _pad_or_truncate(self, data, target_size=(256, 64)):
        target_h, target_w = target_size
        h, w = data.shape

        # freq dimension
        if h < target_h:
            pad_h = target_h - h
            data = np.pad(data, ((0, pad_h), (0, 0)), mode='constant')
        elif h > target_h:
            data = data[:target_h, :]

        # time dimension
        if w < target_w:
            pad_w = target_w - w
            data = np.pad(data, ((0, 0), (0, pad_w)), mode='constant')
        elif w > target_w:
            data = data[:, :target_w]

        return data
