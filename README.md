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
