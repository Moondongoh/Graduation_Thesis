import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
import librosa
from tqdm import tqdm


def extract_wavelet_statistical_features(
    signal, wavelet="db4", level=5, segment_length=512
):
    """
    1D 신호를 세그먼트로 분할하고, 각 세그먼트의 웨이블릿 계수로부터
    통계적 특징을 추출합니다.
    """
    num_segments = len(signal) // segment_length
    if num_segments == 0:
        segments = [signal]
    else:
        segments = [
            signal[i * segment_length : (i + 1) * segment_length]
            for i in range(num_segments)
        ]

    coeff_names = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    feature_columns = []
    for coeff_name in coeff_names:
        for stat in ["mean", "std", "max", "min", "median", "energy"]:
            feature_columns.append(f"{coeff_name}_{stat}")

    features = []
    for segment in segments:
        coeffs = pywt.wavedec(segment, wavelet, level=level)

        segment_features = []
        for c in coeffs:
            stats = [
                np.mean(c),
                np.std(c),
                np.max(c),
                np.min(c),
                np.median(c),
                np.sum(c**2),  # 에너지 << 이제 킥
            ]
            segment_features.extend(stats)
        features.append(segment_features)

    return pd.DataFrame(features, columns=feature_columns)


def process_folder_to_csv(
    folder_path,
    output_csv_path,
    wavelet="db4",
    level=5,
    segment_length=512,
    reference_csv=None,
):
    """
    폴더 안의 모든 WAV 파일을 불러와 웨이블릿 특징을 추출하고
    하나의 CSV로 저장합니다.
    """
    src = Path(folder_path)
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ref_map = {}
    if reference_csv:
        ref_path = Path(reference_csv)
        if ref_path.exists():
            ref_df = pd.read_csv(ref_path, header=None, names=["filename", "label"])
            ref_df["filename"] = ref_df["filename"].astype(str)
            ref_df["label"] = (
                ref_df["label"].map({-1: 0, 1: 1}).fillna(ref_df["label"]).astype(int)
            )
            ref_map = {
                Path(f).stem: int(l)
                for f, l in zip(ref_df["filename"], ref_df["label"])
            }
            print(f"REFERENCE 로드: {len(ref_map)} 항목")
        else:
            print(
                f"WARNING: reference_csv 경로가 없습니다: {reference_csv}  -> 라벨 없이 저장합니다."
            )

    wav_files = sorted([p for p in src.iterdir() if p.suffix.lower() == ".wav"])
    if not wav_files:
        print(f"❌ '{src}' 폴더에 wav 파일이 없습니다.")
        return

    all_features = []
    for wav_path in tqdm(wav_files, desc="Processing WAV files"):
        try:
            y, sr = librosa.load(str(wav_path), sr=None, mono=True)
            df_features = extract_wavelet_statistical_features(
                y, wavelet, level, segment_length
            )
            label = ref_map.get(wav_path.stem, np.nan) if ref_map else np.nan
            df_features["label"] = label
            all_features.append(df_features)
        except Exception as e:
            print(f"⚠️ {wav_path.name} 처리 중 오류 발생: {e}")

    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        cols = [c for c in final_df.columns if c != "file" and c != "sr"]
        save_df = final_df[cols]
        save_df.to_csv(out.with_suffix(".csv"), index=False)
        print(
            f"✅ 모든 특징 추출 완료, CSV 저장: '{out.with_suffix('.csv')}'  (rows: {len(save_df)})"
        )
    else:
        print("❌ 추출된 특징이 없습니다.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Physical_copy Dataset의 wav에 대해 웨이블릿 특징 추출 (features + label 저장)"
    )
    p.add_argument(
        "--src-dir",
        default=r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\Dataset",
        help="입력 WAV 폴더",
    )
    p.add_argument(
        "--out-csv",
        default=r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\Wavelet\wavelet_features.csv",
        help="출력 CSV 경로",
    )
    p.add_argument(
        "--reference-csv",
        default=r"D:\MDO\heartbeat\1_New_HB_0818\balance_data\GAN\Dataset\REFERENCE2_gan_balanced.csv",
        help="REFERENCE CSV 경로 (선택, filename,label 포맷)",
    )
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--segment-length", type=int, default=512)
    args = p.parse_args()

    process_folder_to_csv(
        args.src_dir,
        args.out_csv,
        wavelet=args.wavelet,
        level=args.level,
        segment_length=args.segment_length,
        reference_csv=args.reference_csv,
    )
