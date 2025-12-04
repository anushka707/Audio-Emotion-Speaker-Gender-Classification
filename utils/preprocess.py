import os
import numpy as np
import soundfile as sf
from scipy.signal import resample
from transformers import Wav2Vec2FeatureExtractor

TARGET_SR = 16000
extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/wav2vec2-base",
    return_attention_mask=False
)

# ------------------------------------------
# SAFE AUDIO LOADER (NO TORCHAUDIO)
# ------------------------------------------
def load_audio(path):
    wav, sr = sf.read(path)

    # convert stereo → mono
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)

    # resample to exactly 16k
    if sr != TARGET_SR:
        duration = wav.shape[0] / sr
        new_len = int(TARGET_SR * duration)
        wav = resample(wav, new_len)
        sr = TARGET_SR

    return wav.astype(np.float32), sr


# ------------------------------------------
# WAV → Wav2Vec2 feature extractor
# ------------------------------------------
def wav_to_features(wav, sr):
    feats = extractor(
        wav,
        sampling_rate=sr,
        return_tensors="pt"
    )["input_values"][0]
    return feats


# ------------------------------------------
# RAVDESS filename → emotion
# ------------------------------------------
def ravdess_emotion_from_filename(fname):
    # Filename format: 03-01-05-02-02-02-12.wav
    emo_id = int(fname.split("-")[2])
    mapping = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return mapping.get(emo_id, "unknown")


# ------------------------------------------
# TESS filename → emotion
# ------------------------------------------
def tess_emotion_from_filename(fname):
    # Example: OAF_back_angry.wav
    emo = fname.split("_")[-1].replace(".wav", "")
    if emo == "ps":
        emo = "surprised"
    return emo


# ------------------------------------------
# Load full emotion dataset (RAVDESS + TESS)
# ------------------------------------------
def load_emotion_dataset():
    X_list = []
    y_list = []

    # ---------- RAVDESS ----------
    ravdess_root = "data/ravdess"
    for actor in sorted(os.listdir(ravdess_root)):
        actor_path = os.path.join(ravdess_root, actor)
        if not os.path.isdir(actor_path):
            continue

        for f in sorted(os.listdir(actor_path)):
            if not f.lower().endswith(".wav"):
                continue

            full = os.path.join(actor_path, f)
            wav, sr = load_audio(full)
            feats = wav_to_features(wav, sr)
            X_list.append(feats)
            y_list.append(ravdess_emotion_from_filename(f))

    # ---------- TESS ----------
    tess_root = "data/tess"
    for f in sorted(os.listdir(tess_root)):
        if not f.lower().endswith(".wav"):
            continue

        full = os.path.join(tess_root, f)
        wav, sr = load_audio(full)
        feats = wav_to_features(wav, sr)
        X_list.append(feats)
        y_list.append(tess_emotion_from_filename(f))

    return X_list, y_list
# ------------------------------------------
# Gender dataset loader (RAVDESS + TESS)
# ------------------------------------------
from collections import Counter

def load_gender_dataset():
    X_list = []
    y_list = []

    # ---------- RAVDESS ----------
    ravdess_root = "data/ravdess"
    for actor in sorted(os.listdir(ravdess_root)):
        actor_path = os.path.join(ravdess_root, actor)
        if not os.path.isdir(actor_path):
            continue

        actor_id = int(actor.split("_")[1])
        gender = 1 if actor_id % 2 == 0 else 0  # female if even

        for f in sorted(os.listdir(actor_path)):
            if not f.lower().endswith(".wav"):
                continue

            full = os.path.join(actor_path, f)
            wav, sr = load_audio(full)
            feats = wav_to_features(wav, sr)
            X_list.append(feats)
            y_list.append(gender)

    # ---------- TESS ----------
    tess_root = "data/tess"
    for f in sorted(os.listdir(tess_root)):
        if not f.lower().endswith(".wav"):
            continue
        
        full = os.path.join(tess_root, f)
        wav, sr = load_audio(full)
        feats = wav_to_features(wav, sr)
        X_list.append(feats)

        # fix gender mapping — 100% accurate
        prefix = f[:3].lower()
        if prefix in ["oaf", "yaf"]:  
            y_list.append(1)  # female
        elif prefix in ["oah", "yah"]:
            y_list.append(0)  # male
        else:
            y_list.append(0)

    # compute class weights inside loader
    counts = Counter(y_list)
    male_count = counts[0]
    female_count = counts[1]
    total = male_count + female_count

    class_weights = {
        "male": total / male_count,
        "female": total / female_count
    }

    return X_list, y_list, class_weights

