import librosa
import os

clean_dir = './data/train/clean'
n_fft = 1024
hop_length = 512
sample_rate = 16000

lengths = []
for f in os.listdir(clean_dir):
    if f.endswith('.wav'):
        audio, _ = librosa.load(os.path.join(clean_dir, f), sr=sample_rate)
        spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        lengths.append(spec.shape[1])

print(
    f"Min length: {min(lengths)}, Max length: {max(lengths)}, Average length: {sum(lengths)/len(lengths)}")
