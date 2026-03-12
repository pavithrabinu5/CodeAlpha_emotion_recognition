"""Evaluation utilities for DeepEmotion models."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support, roc_curve, auc
from torch.utils.data import DataLoader

from src.train import FeatureDataset, EMOTION_ORDER, build_model


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(ckpt["model_type"], num_classes=len(ckpt["label_order"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def plot_confusion(y_true, y_pred, labels, out_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate(metadata: str, checkpoint: str, feature_type: str, batch_size: int, out_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(checkpoint, device)

    df = pd.read_csv(metadata)
    df = df[df["split"] == "test"]
    df = df[df["feature_type"] == feature_type]

    loader = DataLoader(FeatureDataset(df, feature_type), batch_size=batch_size, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)[0] if hasattr(model, "attention") else model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    labels = list(range(len(EMOTION_ORDER)))
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=labels, average="macro")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion(all_labels, all_preds, labels=labels, out_path=out_dir / "confusion_matrix.png")

    # ROC curves (one-vs-rest)
    all_probs = np.array(all_probs)
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(EMOTION_ORDER):
        fpr, tpr, _ = roc_curve(np.array(all_labels) == i, all_probs[:, i])
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png")
    plt.close()

    report = classification_report(all_labels, all_preds, target_names=EMOTION_ORDER, digits=3)
    metrics_txt = out_dir / "metrics.txt"
    with open(metrics_txt, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision (macro): {precision:.4f}\n")
        f.write(f"Recall (macro): {recall:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")
        f.write("\nClassification Report\n")
        f.write(report)
    print(f"Saved metrics to {metrics_txt}")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepEmotion model")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--feature_type", type=str, choices=["mel", "mfcc"], default="mel")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    evaluate(args.metadata, args.checkpoint, args.feature_type, args.batch_size, args.out_dir)
