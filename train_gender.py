import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from utils.preprocess import load_gender_dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16
EPOCHS = 10

class GenderNet(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max_len, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)   # male/female
        )

    def forward(self, x):
        return self.net(x)

def train_model():
    print("Loading gender dataset…")
    X_list, y_list, class_weights_dict = load_gender_dataset()

    # compute max length
    max_len = max(x.shape[0] for x in X_list)

    # pad
    X = torch.zeros(len(X_list), max_len)
    for i, feat in enumerate(X_list):
        X[i, :feat.shape[0]] = feat

    y = torch.tensor(y_list, dtype=torch.long)

    # convert class weights to tensor
    class_weights = torch.tensor([
        class_weights_dict["male"],
        class_weights_dict["female"]
    ], dtype=torch.float32).to(DEVICE)

    # weighted loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # sampler to oversample minority class
    sample_weights = [class_weights_dict["male"] if label == 0 else class_weights_dict["female"]
                      for label in y_list]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH, sampler=sampler)

    # model
    model = GenderNet(max_len).to(DEVICE)
    optimzr = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting training…")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch_X, batch_y in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimzr.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimzr.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} — Loss: {total_loss / len(loader):.4f}")

    print("Saving model…")
    torch.save({
        "state_dict": model.state_dict(),
        "max_len": max_len
    }, "models/gender_model.pth")

    print("Training complete!")

if __name__ == "__main__":
    train_model()
