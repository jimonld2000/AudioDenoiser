import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[63, 32, 16], hop_lengths=[16, 8, 4]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_lengths = hop_lengths

    def forward(self, pred, target):
        # If input is 4D ([batch, 1, freq, time]), average over the frequency dimension.
        if pred.dim() == 4:
            pred = pred.mean(dim=2)  # now [batch, 1, time]
            target = target.mean(dim=2)
        # If still 3D with a singleton channel, squeeze it to get [batch, time].
        if pred.dim() == 3 and pred.size(1) == 1:
            pred = pred.squeeze(1)
            target = target.squeeze(1)

        loss = 0.0
        for fft, hop in zip(self.fft_sizes, self.hop_lengths):
            # Create an explicit rectangular window to suppress warnings.
            window = torch.ones(fft, device=pred.device)
            pred_mag = torch.abs(
                torch.stft(pred, n_fft=fft, hop_length=hop, return_complex=True,
                           pad_mode='constant', window=window)
            )
            target_mag = torch.abs(
                torch.stft(target, n_fft=fft, hop_length=hop, return_complex=True,
                           pad_mode='constant', window=window)
            )
            loss += F.l1_loss(pred_mag, target_mag)
        return loss / len(self.fft_sizes)

class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=8000, n_mels=64, n_fft=63, hop_length=16):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    def forward(self, pred, target):
        # If input is 4D ([batch, 1, freq, time]), average over the frequency dimension.
        if pred.dim() == 4:
            pred = pred.mean(dim=2)  # now shape [batch, 1, time]
            target = target.mean(dim=2)
        # If still 3D with a singleton channel, squeeze it to get [batch, time]
        if pred.dim() == 3 and pred.size(1) == 1:
            pred = pred.squeeze(1)
            target = target.squeeze(1)
            
        # Ensure the mel transform is on the same device as the input.
        self.mel_transform = self.mel_transform.to(pred.device)
        
        # Process each sample individually.
        pred_mels = []
        target_mels = []
        for i in range(pred.shape[0]):
            # Each sample: shape (time,) -> add channel dimension: (1, time)
            pred_sample = pred[i].unsqueeze(0)
            target_sample = target[i].unsqueeze(0)
            pred_mels.append(self.mel_transform(pred_sample))
            target_mels.append(self.mel_transform(target_sample))
        pred_mels = torch.stack(pred_mels)  # shape: (batch, n_mels, time_frames)
        target_mels = torch.stack(target_mels)
        
        return torch.nn.functional.l1_loss(pred_mels, target_mels)

class CombinedPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_loss = MultiScaleSTFTLoss()
        self.mel_loss = MelSpectrogramLoss()
        self.l1_loss = nn.L1Loss()
        
        # Loss weights
        self.w_stft = 0.4
        self.w_mel = 0.4
        self.w_l1 = 0.2

    def forward(self, pred, target):
        stft_loss = self.stft_loss(pred, target)
        mel_loss = self.mel_loss(pred, target)
        l1_loss = self.l1_loss(pred, target)
        
        # Weighted combination of losses
        total_loss = (
            self.w_stft * stft_loss + 
            self.w_mel * mel_loss + 
            self.w_l1 * l1_loss
        )
        
        return total_loss, stft_loss, mel_loss, l1_loss
