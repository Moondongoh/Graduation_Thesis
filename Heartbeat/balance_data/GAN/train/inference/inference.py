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

# 설정: 필요에 맞게 경로 수정
WAVELET_CSV = r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\Physical_copy\Wavelet\wavelet_features.csv"
MODEL_PATH = r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\model\model_conv1d_from_wavelet.pt"
SCALER_PATH = r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\model\model_conv1d_from_wavelet.scaler.pkl"  # 없으면 None
OUT_PRED_CSV = Path(MODEL_PATH).with_name(Path(MODEL_PATH).stem + "_preds.csv")
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 모델 구조: 학습 때 사용한 것과 동일해야 함
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

    # 숫자형 컬럼만 사용 (file/sr 등 비수치 컬럼 자동 제외)
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
    Xt = torch.tensor(X).unsqueeze(1)  # (N,1,features)
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
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print(
            "\nClassification Report:\n",
            classification_report(y_true, y_pred, digits=4),
        )
        try:
            print("AUC:", roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        print("Accuracy:", accuracy_score(y_true, y_pred))
    else:
        print("라벨 없음: 예측 결과만 저장합니다.")

    out_df.to_csv(OUT_PRED_CSV, index=True)
    print("예측 결과 저장:", OUT_PRED_CSV)


if __name__ == "__main__":
    infer()
