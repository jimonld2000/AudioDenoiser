import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class MultiScaleSTFTLoss(nn.Module):
    """
    Calculates L1 loss on the magnitude of STFTs with multiple resolutions.
    """
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_lengths=[256, 512, 128]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_lengths = hop_lengths
        assert len(fft_sizes) == len(hop_lengths), "fft_sizes and hop_lengths must have the same length."

    def forward(self, pred_spec, target_spec):
        # This loss operates on spectrograms directly
        loss = 0.0
        
        # Assuming pred_spec and target_spec are log-magnitude spectrograms
        # Convert them back to linear magnitude for L1 loss calculation
        pred_mag = torch.expm1(pred_spec)
        target_mag = torch.expm1(target_spec)
        
        # Simple L1 loss on the input spectrogram resolution
        loss += F.l1_loss(pred_mag, target_mag)

        # Note: A true multi-scale loss would re-calculate STFT from the waveform.
        # This implementation uses a simplified L1 on the provided spectrograms.
        # For a full implementation, you would pass waveforms and calculate STFTs here.
        return loss

class MelSpectrogramLoss(nn.Module):
    """
    Calculates L1 loss on Mel spectrograms.
    """
    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    def forward(self, pred_wav, target_wav):
        # This loss requires waveforms as input
        self.mel_transform = self.mel_transform.to(pred_wav.device)
        
        pred_mels = self.mel_transform(pred_wav.squeeze(1))
        target_mels = self.mel_transform(target_wav.squeeze(1))
        
        # Use log-mel spectrograms for perceptually relevant loss
        log_pred_mels = torch.log1p(pred_mels)
        log_target_mels = torch.log1p(target_mels)

        return F.l1_loss(log_pred_mels, log_target_mels)

class CombinedPerceptualLoss(nn.Module):
    """
    Combined loss for U-Net, operating on magnitude spectrograms.
    This is a simplified version for spectrogram-to-spectrogram models.
    """
    def __init__(self):
        super().__init__()
        # We will use a simple L1 loss on the magnitude spectrograms, 
        # as this is the most direct approach for a spec-to-spec model.
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred_spec, target_spec):
        # pred_spec and target_spec are the output/target from the dataloader
        
        # L1 Loss on the log-magnitude spectrogram
        l1_loss = self.l1_loss(pred_spec, target_spec)
        
        # The other losses are kept for reference but are harder to integrate
        # into a model that doesn't output a waveform. We'll return them as 0.
        stft_loss_val = 0.0 
        mel_loss_val = 0.0
        
        total_loss = l1_loss
        
        return total_loss, l1_loss, stft_loss_val, mel_loss_val
