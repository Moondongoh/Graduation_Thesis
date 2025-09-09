import argparse
import random
import shutil
from pathlib import Path
from uuid import uuid4
from collections import Counter

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import sys

SR = 22050  # 샘플링 레이트
DUR = 1.0  # 지속 시간
NZ = 16  # 잠재 공간 차원

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_wav(name):
    p = Path(name)
    return name if p.suffix else f"{name}.wav"


def unique_dest(dest_dir: Path, base_name: str):
    candidate = dest_dir / base_name
    if not candidate.exists():
        return candidate
    stem = Path(base_name).stem
    suffix = Path(base_name).suffix or ".wav"
    return dest_dir / f"{stem}_copy_{uuid4().hex[:8]}{suffix}"


def extract_rms_mean(wav_path, sr=SR):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    rms = librosa.feature.rms(y=y)
    return float(np.mean(rms))


# 1D GAN 모델
class Generator(nn.Module):
    def __init__(self, nz=NZ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nz, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_gan_on_1d(data_1d, epochs=2000, batch_size=16, nz=NZ, lr=1e-3, verbose=False):

    X = data_1d.reshape(-1, 1).astype(np.float32)

    mu = float(X.mean())
    sigma = float(X.std())
    if sigma == 0:
        sigma = 1e-6
    Xs = (X - mu) / sigma

    dataset = torch.from_numpy(Xs).to(DEVICE)
    n = dataset.shape[0]

    G = Generator(nz=nz).to(DEVICE)
    D = Discriminator().to(DEVICE)
    criterion = nn.BCELoss()
    optD = optim.Adam(D.parameters(), lr=lr)
    optG = optim.Adam(G.parameters(), lr=lr)

    for epoch in range(epochs):

        idx = np.random.randint(0, n, size=min(batch_size, n))
        real = dataset[idx]
        real_labels = torch.ones((real.size(0), 1), device=DEVICE)
        fake_labels = torch.zeros((real.size(0), 1), device=DEVICE)

        D.zero_grad()
        out_real = D(real)
        loss_real = criterion(out_real, real_labels)

        z = torch.randn((real.size(0), nz), device=DEVICE)
        fake = G(z).detach()
        out_fake = D(fake)
        loss_fake = criterion(out_fake, fake_labels)

        lossD = loss_real + loss_fake
        lossD.backward()
        optD.step()

        G.zero_grad()
        z2 = torch.randn((real.size(0), nz), device=DEVICE)
        gen = G(z2)
        out_gen = D(gen)
        lossG = criterion(out_gen, real_labels)
        lossG.backward()
        optG.step()

        if verbose and (epoch % (epochs // 5 + 1) == 0 or epoch < 10):
            print(
                f"Epoch {epoch+1}/{epochs}  lossD:{lossD.item():.4f}  lossG:{lossG.item():.4f}"
            )

    return G.cpu(), float(mu), float(sigma)


def synthesize_wave_from_rms(target_rms, sr=SR, dur=DUR):
    n = int(sr * dur)
    y = np.random.normal(0, 1.0, size=n).astype(np.float32)
    cur_rms = np.sqrt(np.mean(y**2))
    if cur_rms == 0:
        cur_rms = 1e-6
    y = y * (target_rms / cur_rms)
    maxv = np.max(np.abs(y))
    if maxv > 1.0:
        y = y / maxv
    return y


def gan_balance(
    train_dir,
    csv_path,
    out_dir,
    epochs=2000,
    batch_size=16,
    nz=NZ,
    verbose=False,
    dry_run=False,
):
    random.seed(42)
    train_dir = Path(train_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, header=None, names=["filename", "label"])
    df["filename"] = df["filename"].astype(str)

    counts = df["label"].value_counts()
    max_count = counts.max()
    print("원본 라벨 분포:", counts.to_dict())
    print("목표 개수(라벨별):", int(max_count))

    sources = {}
    for label in counts.index:
        names = df[df["label"] == label]["filename"].tolist()
        paths = []
        for n in names:
            p = train_dir / ensure_wav(n)
            if p.exists():
                paths.append(p)
            else:
                print(f"경고: 파일 없음 {p}")
        sources[label] = paths

    out_records = []

    for label, paths in sources.items():
        if not paths:
            print(f"라벨 {label}에 존재하는 wav 파일이 하나도 없습니다. 건너뜁니다.")
            continue

        for src in paths:
            dest_path = unique_dest(out_dir, src.name)
            if dry_run:
                print(f"[DRY RUN] 복사: {src} -> {dest_path.name}")
            else:
                shutil.copy2(src, dest_path)
            out_records.append({"filename": dest_path.stem, "label": label})

        current = len([r for r in out_records if r["label"] == label])
        need = int(max_count) - current
        print(f"라벨 {label}: 현재 {current}, 추가로 생성 필요 {need}")

        if need <= 0:
            continue

        feats = []
        for p in paths:
            try:
                feats.append(extract_rms_mean(p))
            except Exception as e:
                print(f"경고: {p} 로부터 특징 추출 실패:", e)
        feats = np.array(feats, dtype=np.float32)
        if feats.size < 2:
            print(
                f"라벨 {label} 데이터가 너무 적어 GAN 학습 불가. 기존 파일 복제로 대체."
            )
            for _ in range(need):
                src = random.choice(paths)
                dest_path = unique_dest(out_dir, src.name)
                if dry_run:
                    print(f"[DRY RUN] 복제: {src} -> {dest_path.name}")
                else:
                    shutil.copy2(src, dest_path)
                out_records.append({"filename": dest_path.stem, "label": label})
            continue

        print(f"라벨 {label}에 대해 GAN 학습 시작 (epochs={epochs}) ...")
        G, mu, sigma = train_gan_on_1d(
            feats, epochs=epochs, batch_size=batch_size, nz=nz, lr=1e-3, verbose=verbose
        )

        G.eval()
        with torch.no_grad():
            gen_count = 0
            tries = 0

            while gen_count < need and tries < need * 10:
                z = torch.randn((1, nz))
                out_arr = G(z).cpu().numpy().reshape(-1)
                out = float(out_arr[0])
                val = out * sigma + mu
                val = float(np.clip(val, 0.0, None))
                wav = synthesize_wave_from_rms(val, sr=SR, dur=DUR)
                fname = f"gan_{label}_{uuid4().hex[:8]}.wav"
                dest_path = out_dir / fname
                if dry_run:
                    print(f"[DRY RUN] 생성(가상): RMS={val:.6f} -> {fname}")
                else:
                    sf.write(dest_path, wav, SR)
                out_records.append({"filename": Path(fname).stem, "label": label})
                gen_count += 1
                tries += 1

            if gen_count < need:
                print(
                    f"주의: 라벨 {label}에 대해 필요한 생성 개수 {need} 중 {gen_count}만 생성됨."
                )

    if not out_records:
        print("추가된 파일 없음. 종료.")
        return

    balanced_df = pd.DataFrame(out_records)
    out_csv = Path(out_dir) / (Path(csv_path).stem + "_gan_balanced.csv")
    balanced_df.to_csv(out_csv, index=False, header=False)
    print("GAN 복사/생성 완료. 출력 폴더:", out_dir)
    print("생성된 CSV:", out_csv)
    print("최종 라벨 분포:", dict(Counter(balanced_df["label"])))


def report_counts(train_dir, csv_path, out_dir):
    train_dir = Path(train_dir)
    out_dir = Path(out_dir)
    if not Path(csv_path).exists():
        print("ERROR: CSV 파일을 찾을 수 없습니다:", csv_path)
        return

    df = pd.read_csv(csv_path, header=None, names=["filename", "label"])
    df["filename"] = df["filename"].astype(str)
    csv_counts = df["label"].value_counts().sort_index()
    print("CSV 기준 라벨 분포:")
    for lbl, cnt in csv_counts.items():
        print(f"  라벨 {lbl}: {cnt}")

    train_wavs = list(train_dir.glob("*.wav"))
    print(f"\n{train_dir} 폴더의 .wav 파일 개수: {len(train_wavs)}")

    if out_dir.exists():
        out_wavs = list(out_dir.glob("*.wav"))
        print(f"{out_dir} 폴더의 .wav 파일 개수: {len(out_wavs)}")
        try:
            out_df = pd.read_csv(
                Path(out_dir) / (Path(csv_path).stem + "_gan_balanced.csv"),
                header=None,
                names=["filename", "label"],
            )
            out_counts = out_df["label"].value_counts().sort_index()
            print("\nout-dir 기준 라벨 분포 (이미 생성된 경우):")
            for lbl, cnt in out_counts.items():
                print(f"  라벨 {lbl}: {cnt}")
        except Exception:
            pass
    else:
        print(f"{out_dir} 폴더가 존재하지 않습니다.")

    max_count = int(df["label"].value_counts().max())
    print(f"\n목표 개수(라벨별): {max_count}")
    for lbl in sorted(df["label"].unique()):
        current = int(df[df["label"] == lbl].shape[0])
        need = max_count - current
        print(f"  라벨 {lbl}: 현재 {current}, 추가 필요 {max(0, need)}")

    return


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-dir", default=r"D:\MDO\heartbeat\1_New_HB_0818\Dataset\train"
    )
    p.add_argument("--csv", default=None, help="기본: train-dir/REFERENCE2.csv")
    p.add_argument(
        "--out-dir", default=r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\Dataset"
    )
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--nz", type=int, default=NZ)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    csv_path = args.csv or str(Path(args.train_dir) / "REFERENCE2.csv")
    if not Path(csv_path).exists():
        print("ERROR: CSV 파일을 찾을 수 없습니다:", csv_path)
    else:
        gan_balance(
            args.train_dir,
            csv_path,
            args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            nz=args.nz,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )
        report_counts(args.train_dir, csv_path, args.out_dir)
