import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
)

WAVELET_CSV = (
    r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\Wavelet\wavelet_features.csv"
)
MODEL_PATH = r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\model\model_conv1d_from_wavelet.pt"
SCALER_PATH = r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\model\model_conv1d_from_wavelet.scaler.pkl"  # 없으면 None
OUT_PRED_CSV = Path(MODEL_PATH).with_name(Path(MODEL_PATH).stem + "_preds.csv")
BATCH_SIZE = 64
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


def load_features(csv_path, scaler_path=None):
    df = pd.read_csv(csv_path)
    has_label = "label" in df.columns
    if has_label:
        df_valid = df.dropna(subset=["label"]).copy()
        true_labels = df_valid["label"].astype(int).values
        features_df = df_valid.drop(columns=["label"])
    else:
        true_labels = None
        features_df = df.copy()

    feat_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns]
    if not feat_cols:
        raise SystemExit("ERROR: 숫자형 특징 컬럼이 없습니다.")
    X = features_df[feat_cols].values.astype(np.float32)

    if scaler_path and Path(scaler_path).exists():
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    return X, feat_cols, true_labels, df.index if not has_label else df_valid.index


def infer():
    X, feat_cols, true_labels, original_idx = load_features(WAVELET_CSV, SCALER_PATH)
    Xt = torch.tensor(X).unsqueeze(1)
    loader = DataLoader(
        TensorDataset(Xt, torch.zeros(len(Xt), dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = ConvFeatNet(in_channels=1).to(DEVICE)
    if not Path(MODEL_PATH).exists():
        raise SystemExit("ERROR: 모델 파일이 없습니다: " + MODEL_PATH)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    preds = []
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            out = model(xb)
            prob_pos = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            probs.extend(prob_pos.tolist())
            preds.extend(pred.tolist())

    out_df = pd.DataFrame(
        {"pred": np.array(preds), "prob_pos": np.array(probs)}, index=original_idx
    )

    if true_labels is not None:
        out_df["true"] = true_labels
        y_true = out_df["true"].values
        y_pred = out_df["pred"].values
        y_prob = out_df["prob_pos"].values

        print("Samples:", len(y_true))
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:\n", cm)
        print(
            "\nClassification Report:\n",
            classification_report(y_true, y_pred, digits=4),
        )
        try:
            auc = roc_auc_score(y_true, y_prob)
            print(f"AUC: {auc:.4f}")
        except Exception:
            pass
        print("Accuracy:", accuracy_score(y_true, y_pred))

        if cm.size == 4:
            TN, FP, FN, TP = cm.ravel()
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            print(f"Sensitivity (민감도): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
            print(f"Specificity (특이도): {specificity:.4f} ({specificity*100:.2f}%)")
            out_df["true_pos"] = TP
            out_df["true_neg"] = TN
        else:
            print(
                "이진 분류가 아니거나 confusion matrix 형식이 예상과 다릅니다. Sensitivity/Specificity 계산 불가."
            )
    else:
        print("라벨 없음: 예측 결과만 저장합니다.")

    out_df.to_csv(OUT_PRED_CSV, index=True)
    print("예측 결과 저장:", OUT_PRED_CSV)


if __name__ == "__main__":
    infer()
