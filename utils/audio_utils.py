import librosa
import numpy as np

def load_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return y, sr

def audio_to_melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

def normalize_spectrogram(mel_db):
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    return mel_norm

def pad_spectrogram(spec, max_time_steps=300):
    if spec.shape[1] < max_time_steps:
        pad_width = max_time_steps - spec.shape[1]
        spec = np.pad(spec, ((0,0), (0, pad_width)), mode="constant")
    else:
        spec = spec[:, :max_time_steps]
    return spec

def extract_features(file_path, max_time_steps=300):
    y, sr = load_audio(file_path)
    mel = audio_to_melspectrogram(y, sr)
    mel_norm = normalize_spectrogram(mel)
    mel_final = pad_spectrogram(mel_norm, max_time_steps)
    return mel_final

def one_hot(index, num_classes):
    vec = np.zeros(num_classes)
    vec[index] = 1
    return vec
