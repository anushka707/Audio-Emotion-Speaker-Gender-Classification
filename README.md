Real-Time Speech Emotion & Gender Recognition

A deep-learning based system that predicts emotion (8 classes) and gender (male/female) from voice in real time using Wav2Vec2 feature extraction, PyTorch classifiers, and microphone streaming.

This project includes:

• Emotion classification → neutral, calm, happy, sad, angry, fearful, disgust, surprised
• Gender classification → male, female
• Fully trained PyTorch models
• Real-time microphone inference (realtime_predict.py)
• Offline prediction for .wav files (predict.py)
• Training scripts for both models
• Evaluation scripts (confusion matrix, ROC curves, PR curves)
• Clean preprocessing pipeline using RAVDESS + TESS datasets

Project Structure

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


Features
Emotion Recognition

• Trained on RAVDESS + TESS
• 8 emotion classes
• Strong accuracy on clean audio
• Robust to light noise

Gender Recognition

• Trained on balanced male/female samples
• Wav2Vec2 embeddings + Dense classifier
• High accuracy

Real-Time Predictions

• Microphone streaming every 0.5 seconds
• Wav2Vec2 feature extraction per chunk
• Prints live predictions continuously

Installation

Clone the repository:

git clone https://github.com/anushka707/Audio-Emotion-Speaker-Gender-Classification

Create virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt


Training the Models
Train Emotion Model

python train_emotion.py

Model saved to:
models/emotion_model.pth

Train Gender Model

python train_gender.py

Model saved to:
models/gender_model.pth


Evaluation

Generates:
• Confusion Matrix
• Classification Report
• ROC Curves
• Precision–Recall Curves

Run:

python evaluate.py

Outputs saved in:
results/


Offline Prediction

Predict emotion + gender from a .wav file:

python predict.py sample.wav

Example output:

Emotion: happy
Gender: female


Real-Time Microphone Prediction

Start live prediction:

python realtime_predict.py

Example output:

Emotion: excited | Gender: female
Emotion: calm | Gender: female
Emotion: surprised | Gender: female

Stop using:

Ctrl + C


Model Architecture
Feature Extraction

• Uses facebook/wav2vec2-base
• Resamples audio to 16 kHz
• Extracts dense embeddings
• No torchaudio required (soundfile + scipy instead)

Emotion Classifier

• Input: Wav2Vec2 feature vector
• Dense layers
• ReLU activation
• Softmax output (8 classes)

Gender Classifier

• Same idea but fewer layers
• Sigmoid output


Datasets Used
RAVDESS

• 24 actors
• 8 emotions
• Clean studio recordings

TESS

• 2000+ utterances
• Covers similar emotion spectrum
• High-quality samples ideal for ML


Future Enhancements

• ONNX export for deployment
• Streamlit dashboard
• Add noise reduction
• Multi-language support
• Add emotion intensity classification
