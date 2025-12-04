import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.preprocess import load_emotion_dataset


# ---------------------------------------------
# Dataset
# ---------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------
# Collate function for padding
# ---------------------------------------------
def collate_fn(batch):
    sequences = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch])

    max_len = max(seq.shape[0] for seq in sequences)
    padded = torch.zeros(len(sequences), max_len)

    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0]] = seq

    return padded, labels


# ---------------------------------------------
# Simple MLP classifier
# ---------------------------------------------
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


# ---------------------------------------------
# Training loop
# ---------------------------------------------
def train_model():
    print("Loading emotion dataset…")
    X_list, y_list = load_emotion_dataset()

    # Label encoding
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_list)

    # Step 1 — split before padding
    X_train, X_test, y_train, y_test = train_test_split(
        X_list, y_encoded, test_size=0.2, random_state=42
    )

    # Step 2 — compute global max length across ALL samples
    max_len_global = max(seq.shape[0] for seq in X_list)
    print("Global max length:", max_len_global)

    # Step 3 — pad all sequences NOW so model sees consistent input dimension
    def pad_list(seq_list):
        padded = []
        for seq in seq_list:
            pad = torch.zeros(max_len_global)
            pad[:seq.shape[0]] = seq
            padded.append(pad)
        return torch.stack(padded)

    X_train_pad = pad_list(X_train)
    X_test_pad = pad_list(X_test)

    # Dataset + Loader
    train_ds = EmotionDataset(X_train_pad, torch.tensor(y_train))
    test_ds = EmotionDataset(X_test_pad, torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    num_classes = len(set(y_list))

    # Step 4 — input_dim MUST equal max_len_global (guaranteed)
    model = EmotionClassifier(max_len_global, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10
    print("\nStarting training…\n")

    # Training
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100)
        for X_batch, y_batch in loop:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} — Loss: {total_loss/len(train_loader):.4f}")

    print("\nTraining complete!")
    torch.save(model.state_dict(), "emotion_model.pth")
    print("Saved: emotion_model.pth")


if __name__ == "__main__":
    train_model()
