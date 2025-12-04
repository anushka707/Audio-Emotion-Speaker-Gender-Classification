import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc,
    precision_recall_curve
)
from utils.preprocess import load_emotion_dataset, load_gender_dataset

# Ensure results/ exists
os.makedirs("results", exist_ok=True)

# ============================================================
# MODELS (same as training)
# ============================================================

class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class GenderNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# CONFIG
# ============================================================
EMOTION_MODEL_PATH = "models/emotion_model.pth"
GENDER_MODEL_PATH = "models/gender_model.pth"
EMOTION_LABELS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised","ps"]

# Your final padded dimension (printed during train)
INPUT_DIM = 84351
EMOTION_CLASSES = 9


# ============================================================
# HELPER: pad sequences to INPUT_DIM
# ============================================================
def pad(x):
    out = torch.zeros(INPUT_DIM)
    L = min(INPUT_DIM, x.shape[0])
    out[:L] = x[:L]
    return out


# ============================================================
# EVALUATE EMOTION MODEL
# ============================================================
def evaluate_emotion():
    print("\nEvaluating Emotion Model…")

    X_list, y_list = load_emotion_dataset()

    # dynamically find all emotion labels
    unique_labels = sorted(list(set(y_list)))
    num_classes = len(unique_labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    X = torch.stack([pad(x) for x in X_list])
    y = np.array([label_to_idx[lbl] for lbl in y_list])

    model = EmotionClassifier(INPUT_DIM, num_classes)
    model.load_state_dict(torch.load(EMOTION_MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1).numpy()

    # confusion matrix
    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=unique_labels)
    disp.plot(xticks_rotation=45)
    plt.title("Emotion Confusion Matrix")
    plt.savefig("results/emotion_confusion_matrix.png", dpi=300)
    plt.close()

    # classification report
    report = classification_report(y, preds, target_names=unique_labels)
    with open("results/emotion_report.txt", "w") as f:
        f.write(report)

    print("Emotion evaluation saved in results/")



# ============================================================
# EVALUATE GENDER MODEL
# ============================================================
def evaluate_gender():
    print("\nEvaluating Gender Model…")

    X_list, y_list = load_gender_dataset()

    X = torch.stack([pad(x) for x in X_list])
    y = np.array(y_list).astype(int)

    model = GenderNet(INPUT_DIM)
    model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(X).squeeze()
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Female", "Male"])
    disp.plot()
    plt.title("Gender Confusion Matrix")
    plt.savefig("results/gender_confusion_matrix.png", dpi=300)
    plt.close()

    # ---------------- Classification Report ----------------
    report = classification_report(y, preds, target_names=["Female", "Male"])
    with open("results/gender_report.txt", "w") as f:
        f.write(report)

    # ---------------- ROC Curve ----------------
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Gender ROC Curve")
    plt.legend()
    plt.savefig("results/gender_roc_curve.png", dpi=300)
    plt.close()

    # ---------------- Precision-Recall Curve ----------------
    prec, rec, _ = precision_recall_curve(y, probs)
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Gender Precision-Recall Curve")
    plt.savefig("results/gender_pr_curve.png", dpi=300)
    plt.close()

    print("Gender evaluation saved in results/")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    evaluate_emotion()
    evaluate_gender()
    print("\nAll evaluation graphs and reports saved inside results/.")
