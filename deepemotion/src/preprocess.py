"""Dataset preprocessing for DeepEmotion.

Creates a unified metadata CSV, cleans audio, applies optional augmentation,
and writes standardized wav files.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from utils.audio_utils import load_audio, trim_silence, reduce_noise, normalize_signal, pad_or_truncate, save_audio, seed_everything
from src import augment

# Unified label mapping across datasets
LABEL_MAP = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fear",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "boredom": "neutral",
}

SUPPORTED_DATASETS = {"ravdess", "tess", "emodb"}


def parse_emotion_from_filename(dataset: str, file_path: Path) -> str:
    name = file_path.stem.lower()
    if dataset == "tess":
        # Example: OAF_happy_1.wav
        parts = name.split("_")
        return parts[1]
    if dataset == "emodb":
        # Example: 03a01Fa.wav -> 'F' for fear, 'W' for anger, etc.
        code = name[5]
        mapping = {
            "W": "angry",
            "L": "boredom",
            "E": "disgust",
            "A": "fear",
            "F": "happy",
            "T": "sad",
            "N": "neutral",
        }
        return mapping.get(code, "neutral")
    if dataset == "ravdess":
        # Filename format: 03-01-05-02-02-02-12.wav (emotion is third field)
        parts = name.split("-")
        emotion_code = parts[2]
        mapping = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprise",
        }
        return mapping.get(emotion_code, "neutral")
    return "neutral"


def discover_files(data_root: Path) -> List[Tuple[str, Path]]:
    """Walk through dataset folders and return (dataset, path)."""
    pairs: List[Tuple[str, Path]] = []
    for dataset in SUPPORTED_DATASETS:
        ds_path = data_root / dataset
        if not ds_path.exists():
            continue
        for wav in ds_path.rglob("*.wav"):
            pairs.append((dataset, wav))
    return pairs


def standardize_label(raw_label: str) -> str:
    return LABEL_MAP.get(raw_label.lower(), "neutral")


def build_metadata(data_root: str, output_csv: str) -> None:
    """Scan datasets, create metadata CSV with columns: split, file_path, emotion."""
    data_root = Path(data_root)
    rows = []
    files = discover_files(data_root)
    for dataset, path in files:
        raw_label = parse_emotion_from_filename(dataset, path)
        label = standardize_label(raw_label)
        rows.append({"file_path": str(path), "emotion": label})

    # Stratified split (80/10/10)
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.DataFrame(rows)
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["emotion"], random_state=42)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    full_df = pd.concat([train_df, val_df, test_df], axis=0)
    full_df.to_csv(output_csv, index=False)
    print(f"Saved metadata to {output_csv} with {len(full_df)} rows")


def process_and_save(metadata_csv: str, cleaned_dir: str, apply_aug: bool = False, max_duration: float = 4.0) -> None:
    """Clean audio, optionally augment, and save standardized wavs.

    Produces new metadata with paths to cleaned files.
    """
    import pandas as pd

    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    new_rows: List[Dict[str, str]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        audio, sr = load_audio(row["file_path"], target_sr=16000)
        audio = trim_silence(audio)
        audio = reduce_noise(audio, sr)
        audio = normalize_signal(audio)
        audio = pad_or_truncate(audio, sr, max_duration=max_duration)

        base_name = Path(row["file_path"]).stem
        out_path = cleaned_dir / f"{base_name}_clean.wav"
        save_audio(out_path, audio, sr)
        new_rows.append({"split": row["split"], "emotion": row["emotion"], "file_path": str(out_path)})

        if apply_aug and row["split"] == "train":
            for idx, variant in enumerate(augment.apply_augmentations(audio, sr)):
                aug_path = cleaned_dir / f"{base_name}_aug{idx}.wav"
                save_audio(aug_path, variant, sr)
                new_rows.append({"split": "train", "emotion": row["emotion"], "file_path": str(aug_path)})

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(cleaned_dir / "metadata_clean.csv", index=False)
    print(f"Cleaned audio saved to {cleaned_dir} with {len(new_df)} rows")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess emotional speech datasets")
    parser.add_argument("--data_root", type=str, required=True, help="Folder containing ravdess/, tess/, emodb/")
    parser.add_argument("--out_csv", type=str, default="outputs/metadata.csv")
    parser.add_argument("--clean_dir", type=str, default="outputs/clean_wav")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    seed_everything(42)
    build_metadata(args.data_root, args.out_csv)
    process_and_save(args.out_csv, args.clean_dir, apply_aug=args.augment)
