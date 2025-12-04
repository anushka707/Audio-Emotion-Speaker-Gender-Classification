import os
import librosa
import numpy as np
import pickle

SAMPLE_RATE = 16000
N_MELS = 128
HOP = 512
DURATION = 3
MAX_LEN = SAMPLE_RATE * DURATION

EMOTION_MAP = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7
}

def extract_ravdess_label(fname):
    # Example: 03-01-05-01-02-02-12.wav
    emo_code = int(fname.split("-")[2])
    emo_names = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return EMOTION_MAP[emo_names[emo_code]]

def extract_tess_label(folder):
    folder = folder.lower()
    if "angry" in folder: return EMOTION_MAP["angry"]
    if "disgust" in folder: return EMOTION_MAP["disgust"]
    if "fear" in folder: return EMOTION_MAP["fearful"]
    if "happy" in folder: return EMOTION_MAP["happy"]
    if "neutral" in folder: return EMOTION_MAP["neutral"]
    if "pleasant" in folder: return EMOTION_MAP["surprised"]
    if "sad" in folder: return EMOTION_MAP["sad"]
    return None

def preprocess_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    if len(y) < MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)))
    else:
        y = y[:MAX_LEN]

    mel = librosa.feature.melspectrogram(y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP)
    mel = librosa.power_to_db(mel, ref=np.max)

    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-9)
    mel = mel[..., np.newaxis]   # (128, time, 1)
    return mel

def load_all_data():
    X, y = [], []

    # RAVDESS
    for actor in os.listdir("data/RAVDESS"):
        actor_path = os.path.join("data/RAVDESS", actor)
        for f in os.listdir(actor_path):
            if f.endswith(".wav"):
                emo = extract_ravdess_label(f)
                mel = preprocess_audio(os.path.join(actor_path, f))
                X.append(mel)
                y.append(emo)

    # TESS
    for folder in os.listdir("data/TESS"):
        folder_path = os.path.join("data/TESS", folder)
        emo = extract_tess_label(folder)
        for f in os.listdir(folder_path):
            if f.endswith(".wav"):
                mel = preprocess_audio(os.path.join(folder_path, f))
                X.append(mel)
                y.append(emo)

    X = np.array(X)
    y = np.array(y)

    print("Saved shapes:", X.shape, y.shape)
    pickle.dump((X, y), open("features/emotion_full.pkl", "wb"))

load_all_data()
# import librosa
# import librosa.feature

# print("librosa module:", librosa)
# print("librosa path:", librosa.__file__)

# print("librosa.feature module:", librosa.feature)
# print("librosa.feature path:", librosa.feature.__file__)
