
from sklearn.metrics import mean_squared_error
import soundfile as sf
import torch
import numpy as np
import librosa
import sys
import os

# Add the `code` directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'training')))

from model import AudioDenoiserUNet


def load_model(model_path, device):
    model = AudioDenoiserUNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def audio_to_spectrogram(audio, n_fft=1024, hop_length=512):
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    return np.abs(spec)


def spectrogram_to_audio(spec, sr, hop_length=512):
    return librosa.istft(spec, hop_length=hop_length)


def denoise_audio(model, noisy_audio, sr, device, n_fft=1024, hop_length=512):
    noisy_spec = audio_to_spectrogram(
        noisy_audio, n_fft=n_fft, hop_length=hop_length)
    noisy_tensor = torch.tensor(noisy_spec, dtype=torch.float32).unsqueeze(
        0).unsqueeze(0).to(device)

    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)
    denoised_spec = denoised_tensor.squeeze().cpu().numpy()
    return spectrogram_to_audio(denoised_spec, sr, hop_length=hop_length)


def evaluate_model(model, clean_dir, noisy_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    clean_files = sorted(
        [f for f in os.listdir(clean_dir) if f.endswith('.wav')])
    noisy_files = sorted(
        [f for f in os.listdir(noisy_dir) if f.endswith('.wav')])

    mse_scores = []

    for clean_file, noisy_file in zip(clean_files, noisy_files):
        clean_path = os.path.join(clean_dir, clean_file)
        noisy_path = os.path.join(noisy_dir, noisy_file)

        clean_audio, sr = librosa.load(clean_path, sr=16000)
        noisy_audio, _ = librosa.load(noisy_path, sr=16000)

        denoised_audio = denoise_audio(model, noisy_audio, sr, device)
        output_path = os.path.join(output_dir, f"denoised_{noisy_file}")
        sf.write(output_path, denoised_audio, sr)

        # Calculate MSE
        clean_audio = clean_audio[:len(denoised_audio)]
        mse = mean_squared_error(clean_audio, denoised_audio)
        mse_scores.append(mse)

        print(f"{noisy_file}: MSE = {mse:.4f}")

    print(f"Average MSE: {np.mean(mse_scores):.4f}")


if __name__ == "__main__":
    clean_dir = './data/test/clean'
    noisy_dir = './data/test/noisy'
    output_dir = './data/test/denoised'
    model_path = './logs/best_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, device)

    evaluate_model(model, clean_dir, noisy_dir, output_dir, device)
