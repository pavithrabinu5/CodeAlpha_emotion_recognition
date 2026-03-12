"""Inference script for DeepEmotion."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from utils.audio_utils import load_audio, trim_silence, reduce_noise, normalize_signal, pad_or_truncate
from src.feature_extraction import extract_features
from src.train import EMOTION_ORDER, build_model


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["model_type"], num_classes=len(ckpt["label_order"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def prepare_input(audio_path: str, feature_type: str):
    audio, sr = load_audio(audio_path, target_sr=16000)
    audio = trim_silence(audio)
    audio = reduce_noise(audio, sr)
    audio = normalize_signal(audio)
    audio = pad_or_truncate(audio, sr, max_duration=4.0)
    feats = extract_features(audio, sr)

    if feature_type == "mel":
        x = feats["mel"][np.newaxis, np.newaxis, :, :]  # (1,1,mel,time)
    else:
        x = feats["mfcc"].T[np.newaxis, :, :]  # (1,time,feat)
    return torch.tensor(x, dtype=torch.float32)


def predict(audio_path: str, checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(checkpoint, device)
    feature_type = ckpt.get("feature_type", "mel")

    x = prepare_input(audio_path, feature_type).to(device)
    with torch.no_grad():
        logits = model(x)[0] if hasattr(model, "attention") else model(x)
        probs = torch.softmax(logits, dim=1).squeeze()
        conf, pred_idx = torch.max(probs, dim=0)
    emotion = EMOTION_ORDER[pred_idx.item()]
    return emotion, conf.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion for a wav file")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    emo, score = predict(args.audio, args.checkpoint)
    print(f"Predicted Emotion: {emo}")
    print(f"Confidence Score: {score:.2f}")
