# Real-Time Speech Emotion & Gender Recognition

A deep-learning based system that predicts **emotion (8 classes)** and **gender (male/female)** from voice **in real time** using **Wav2Vec2 feature extraction**, **PyTorch classifiers**, and **microphone streaming**.

---

## Features

### Emotion Classification  
Predicts:  
**neutral, calm, happy, sad, angry, fearful, disgust, surprised**

### Gender Classification  
Predicts:  
**male, female**

### System Highlights  
- Fully trained PyTorch models  
- Real-time microphone inference (`realtime_predict.py`)  
- Offline prediction for `.wav` files (`predict.py`)  
- Training scripts for both models  
- Evaluation: confusion matrix, ROC curves, PR curves  
- Clean preprocessing pipeline using **RAVDESS + TESS**  

---

## Project Structure
Project/
│
├── models/
│ ├── emotion_model.pth
│ ├── gender_model.pth
│
├── utils/
│ ├── preprocess.py
│
├── train_emotion.py
├── train_gender.py
├── evaluate.py
├── predict.py
├── realtime_predict.py
│
├── requirements.txt
├── README.md
└── .gitignore

## Installation

git clone https://github.com/anushka707/Audio-Emotion-Speaker-Gender-Classification
cd Audio-Emotion-Speaker-Gender-Classification


### Create virtual environment
python3 -m venv venv
source venv/bin/activate

### Install dependencies
pip install -r requirements.txt

## Training the Models

### Train Emotion Model
python train_emotion.py

saved to: models/emotion_model.pth

### Train Gender Model
python train_gender.py

saved to: models/gender_model.pth

## Evaluation

Generates:
Confusion Matrix
Classification Report
ROC Curves
Precision–Recall Curves

Run:
python evaluate.py

All plots saved in:
results/


## Offline Prediction

Predict emotion + gender from a .wav file:
python predict.py sample.wav


Example output:
Emotion: happy
Gender: female

## Real-Time Microphone Prediction

Start live inference:
python realtime_predict.py

Example output:
Emotion: surprised | Gender: female
Emotion: calm      | Gender: male


## Model Architecture
### Feature Extraction
-Uses facebook/wav2vec2-base
-Audio always resampled to 16 kHz
=Extracts dense embeddings
-Avoids torchaudio → uses soundfile + scipy


### Emotion Classifier
-Input: Wav2Vec2 embedding
-MLP with:
Dense → ReLU → Dropout
Dense → ReLU → Dropout
Softmax output

### Gender Classifier
-Smaller MLP
=Sigmoid output
