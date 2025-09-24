import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

WAVELET_CSV = (
    r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\Wavelet\wavelet_features.csv"
)
OUT_MODEL = r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\model\model_conv1d_from_wavelet.pt"
OUT_SCALER = r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\model\model_conv1d_from_wavelet.scaler.pkl"

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
VAL_SPLIT = 0.1
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvFeatNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_wavelet_csv(path):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise SystemExit("ERROR: CSV에 'label' 컬럼이 필요합니다.")
    df = df.dropna(subset=["label"]).copy()
    if df.empty:
        raise SystemExit("ERROR: label이 채워진 행이 없습니다.")
    df["label"] = df["label"].astype(int)
    feat_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != "label"
    ]
    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int64)
    return X, y, feat_cols


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X, y, feat_cols = load_wavelet_csv(WAVELET_CSV)
    print(
        f"Loaded features: {X.shape}, labels: {y.shape}, features cols: {len(feat_cols)}"
    )

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    Path(OUT_MODEL).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, OUT_SCALER)
    print("Saved scaler:", OUT_SCALER)

    Xt = torch.tensor(Xs).unsqueeze(1)
    yt = torch.tensor(y)

    n = len(yt)
    idx = np.arange(n)
    np.random.shuffle(idx)
    nval = max(1, int(n * VAL_SPLIT))
    val_idx = idx[:nval]
    train_idx = idx[nval:]

    train_ds = TensorDataset(Xt[train_idx], yt[train_idx])
    val_ds = TensorDataset(Xt[val_idx], yt[val_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvFeatNet(in_channels=1).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        vloss = 0.0
        vcorrect = 0
        vtotal = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb)
                vloss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                vcorrect += (preds == yb).sum().item()
                vtotal += xb.size(0)
        val_loss = vloss / max(1, vtotal)
        val_acc = vcorrect / max(1, vtotal)

        print(
            f"Epoch {epoch}/{EPOCHS}  train_loss:{train_loss:.4f} acc:{train_acc:.4f}  val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}"
        )

    torch.save(model.state_dict(), OUT_MODEL)
    print("Saved model:", OUT_MODEL)


if __name__ == "__main__":
    main()
