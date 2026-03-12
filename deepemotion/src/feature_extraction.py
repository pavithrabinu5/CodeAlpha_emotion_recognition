"""Feature extraction for speech emotion recognition."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

from utils.audio_utils import load_audio
from src.augment import spec_augment


FEATURE_TYPES = {"mel", "mfcc"}


def extract_features(audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    chroma = librosa.feature.chroma_stft(S=mel_spec, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)

    features = {
        "mel": mel_db.astype(np.float32),
        "mfcc": np.vstack([mfcc, mfcc_delta]).astype(np.float32),
        "chroma": chroma.astype(np.float32),
        "zcr": zcr.astype(np.float32),
        "centroid": centroid.astype(np.float32),
        "rolloff": rolloff.astype(np.float32),
    }
    return features


def save_feature_arrays(metadata_csv: str, output_dir: str, apply_specaug: bool = False) -> str:
    """Compute and save features as .npy files. Returns path to new metadata CSV."""
    df = pd.read_csv(metadata_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        audio, sr = load_audio(row["file_path"], target_sr=16000)
        feats = extract_features(audio, sr)

        for key in ["mel", "mfcc"]:
            feat = feats[key]
            if apply_specaug and key == "mel" and row["split"] == "train":
                feat = spec_augment(feat)
            fname = f"{Path(row['file_path']).stem}_{key}.npy"
            fpath = out_dir / fname
            np.save(fpath, feat)
            records.append({"split": row["split"], "emotion": row["emotion"], "feature_type": key, "feature_path": str(fpath)})

    out_csv = out_dir / "metadata_features.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved feature metadata to {out_csv}")
    return str(out_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from cleaned audio")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/features")
    parser.add_argument("--specaug", action="store_true")
    args = parser.parse_args()

    save_feature_arrays(args.metadata, args.out_dir, apply_specaug=args.specaug)
