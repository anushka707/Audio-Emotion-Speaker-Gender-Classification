import sounddevice as sd
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from scipy.signal import resample
from transformers import Wav2Vec2FeatureExtractor

# ================================
# Configuration
# ================================
TARGET_SR = 16000
FRAME_SECONDS = 1.0               # 1 second of audio per inference
FRAME_SAMPLES = int(TARGET_SR * FRAME_SECONDS)

EMOTION_MODEL_PATH = "models/emotion_model.pth"
GENDER_MODEL_PATH  = "models/gender_model.pth"

# EXACT LABELS USED DURING TRAINING
EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised", "ps"
]

GENDER_LABELS = ["male", "female"]

extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base",
    return_attention_mask=False
)

device = "cpu"

# ================================
# Audio Loader
# ================================
def load_audio_array(arr, sr):
    """Convert microphone audio chunk → mono → resample → float32"""
    if arr.ndim > 1:
        arr = arr.mean(axis=1)

    if sr != TARGET_SR:
        duration = arr.shape[0] / sr
        new_len = int(TARGET_SR * duration)
        arr = resample(arr, new_len)
        sr = TARGET_SR

    return arr.astype(np.float32), sr


# ================================
# Feature Extraction
# ================================
def wav_to_features(wav, sr):
    feats = extractor(
        wav,
        sampling_rate=sr,
        return_tensors="pt"
    )["input_values"][0]
    return feats


# ================================
# Emotion Net (MATCHES TRAIN MODEL)
# ================================
class EmotionNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)




# ================================
# Gender Net (MATCHES TRAIN MODEL)
# ================================
class GenderNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)   # binary output
        )

    def forward(self, x):
        return self.net(x)



# ================================
# Model Loader
# ================================
def load_model(path, model_class, out_dim):
    print(f"Loading model: {path}")

    state = torch.load(path, map_location="cpu")

    # If checkpoint = {"state_dict": ..., "max_len": ...}
    if isinstance(state, dict) and "state_dict" in state and "max_len" in state:
        max_len = state["max_len"]
        sd = state["state_dict"]
    else:
        # Plain state_dict ONLY
        sd = state

        # Infer input dimension from first linear layer
        first_key = list(sd.keys())[0]            # e.g., "net.0.weight"
        weight = sd[first_key]                    # shape = [H, input_dim]
        max_len = weight.shape[1]                 # input_dim = max_len

        print(f"Inferred max_len = {max_len}")

    # Emotion model → 2 args (input_dim, num_classes)
# Gender model  → 1 arg  (input_dim)
    try:
        model = model_class(max_len, out_dim)
    except TypeError:
        model = model_class(max_len)

    model.load_state_dict(sd)
    model.eval()
    return model, max_len



# ================================
# Real-time Prediction
# ================================
def predict_live():
    print("\n========== LIVE VOICE EMOTION + GENDER RECOGNITION ==========")
    print("Speak into microphone… Ctrl+C to stop.\n")

    # Load both models
    emotion_model, emo_max_len = load_model(EMOTION_MODEL_PATH, EmotionNet, len(EMOTION_LABELS))
    gender_model, gen_max_len  = load_model(GENDER_MODEL_PATH, GenderNet, len(GENDER_LABELS))


    # For storing rolling audio
    stream_buffer = np.zeros(0, dtype=np.float32)

    def audio_callback(indata, frames, time, status):
        nonlocal stream_buffer
        chunk = indata[:, 0]  # first channel only
        stream_buffer = np.concatenate([stream_buffer, chunk])

    # Start microphone stream
    with sd.InputStream(
        channels=1,
        samplerate=TARGET_SR,
        callback=audio_callback
    ):
        while True:
            if len(stream_buffer) >= FRAME_SAMPLES:
                piece = stream_buffer[:FRAME_SAMPLES]
                stream_buffer = stream_buffer[FRAME_SAMPLES:]

                wav, sr = load_audio_array(piece, TARGET_SR)
                feats = wav_to_features(wav, sr)

                # pad/truncate for emotion model
                emo_x = torch.zeros(emo_max_len)
                L = min(emo_max_len, feats.shape[0])
                emo_x[:L] = feats[:L]

                # pad/truncate for gender model
                gen_x = torch.zeros(gen_max_len)
                Lg = min(gen_max_len, feats.shape[0])
                gen_x[:Lg] = feats[:Lg]

                with torch.no_grad():
                    emo_pred = emotion_model(emo_x)
                    gen_pred = gender_model(gen_x)

                emo_label = EMOTION_LABELS[int(torch.argmax(emo_pred).item())]
                gen_label = GENDER_LABELS[int(torch.argmax(gen_pred).item())]

                print(f"Emotion: {emo_label:12s} | Gender: {gen_label}")

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    try:
        predict_live()
    except KeyboardInterrupt:
        print("\nStopped.")