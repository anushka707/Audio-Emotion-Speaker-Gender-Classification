import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
from scipy.signal import resample
from transformers import Wav2Vec2FeatureExtractor

# ============================================================
# LOAD WAV2VEC2 FEATURE EXTRACTOR
# ============================================================
TARGET_SR = 16000
extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base",
    return_attention_mask=False
)


def load_audio(path):
    wav, sr = sf.read(path)

    if wav.ndim > 1:
        wav = wav.mean(axis=-1)

    if sr != TARGET_SR:
        duration = wav.shape[0] / sr
        new_len = int(TARGET_SR * duration)
        wav = resample(wav, new_len)
        sr = TARGET_SR

    return wav.astype(np.float32), sr


def wav_to_features(wav, sr):
    feats = extractor(
        wav,
        sampling_rate=sr,
        return_tensors="pt"
    )["input_values"][0]
    return feats


# ============================================================
# MODEL DEFINITIONS — EXACT SAME AS TRAINING
# ============================================================

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class GenderNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# CONSTANTS (values learned during training)
# ============================================================

EMOTION_MODEL_PATH = "models/emotion_model.pth"
GENDER_MODEL_PATH = "models/gender_model.pth"

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised", "ps"
]

INPUT_DIM = 84351   # <-- from your training print
EMOTION_CLASSES = 9


# ============================================================
# PADDING FUNCTION
# ============================================================
def pad_features(feats, max_len):
    out = torch.zeros(max_len)
    L = min(max_len, feats.shape[0])
    out[:L] = feats[:L]
    return out.unsqueeze(0)  # shape: (1, max_len)


# ============================================================
# PREDICT FUNCTION
# ============================================================
def predict(audio_path):
    print("Loading audio:", audio_path)
    wav, sr = load_audio(audio_path)
    feats = wav_to_features(wav, sr)

    X = pad_features(feats, INPUT_DIM)

    # models
    emotion_model = EmotionClassifier(INPUT_DIM, EMOTION_CLASSES)
    gender_model = GenderNet(INPUT_DIM)

    emotion_model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location="cpu"))
    gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location="cpu"))

    emotion_model.eval()
    gender_model.eval()

    with torch.no_grad():
        emo_logits = emotion_model(X)
        gender_logits = gender_model(X)

    emo_idx = emo_logits.argmax(dim=1).item()
    emotion = EMOTION_LABELS[emo_idx]

    gender_prob = torch.sigmoid(gender_logits).item()
    gender = "female" if gender_prob < 0.5 else "male"

    print("\n===== RESULT =====")
    print("Emotion:", emotion)
    print("Gender:", gender)
    print(f"Gender confidence: {gender_prob:.3f}")
    print("==================\n")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <audiofile>")
        exit(1)

    predict(sys.argv[1])
