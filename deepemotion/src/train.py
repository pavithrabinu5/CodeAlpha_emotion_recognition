"""Training script for DeepEmotion."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.audio_utils import seed_everything
from src.cnn_model import CNNEmotionNet
from src.bilstm_model import BiLSTMEmotionNet
from src.hybrid_model import CNNBiLSTMEmotionNet


EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class FeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_type: str):
        self.df = df.reset_index(drop=True)
        self.feature_type = feature_type
        self.label_to_idx = {emo: i for i, emo in enumerate(EMOTION_ORDER)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        feat = np.load(row.feature_path)
        if self.feature_type == "mel":
            # Shape: (mel_bins, time) -> add channel
            feat = feat[np.newaxis, :, :]
            tensor = torch.tensor(feat, dtype=torch.float32)
        elif self.feature_type == "mfcc":
            # Shape: (features, time) -> (time, features)
            tensor = torch.tensor(feat.T, dtype=torch.float32)
        else:
            raise ValueError("Unsupported feature type")
        label = torch.tensor(self.label_to_idx[row.emotion], dtype=torch.long)
        return tensor, label


def make_loaders(meta_csv: str, feature_type: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    df = pd.read_csv(meta_csv)
    df = df[df["feature_type"] == feature_type]

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # Class weights for imbalance handling
    class_counts = train_df["emotion"].value_counts()
    weights = [1.0 / class_counts[e] for e in train_df["emotion"]]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(FeatureDataset(train_df, feature_type), batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(FeatureDataset(val_df, feature_type), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FeatureDataset(test_df, feature_type), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_model(model_type: str, num_classes: int = 7):
    if model_type == "cnn":
        return CNNEmotionNet(num_classes)
    if model_type == "bilstm":
        return BiLSTMEmotionNet(num_classes=num_classes)
    if model_type == "hybrid":
        return CNNBiLSTMEmotionNet(num_classes=num_classes)
    raise ValueError("Unsupported model type")


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)[0] if isinstance(model, BiLSTMEmotionNet) else model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    writer = SummaryWriter(log_dir=args.log_dir)

    train_loader, val_loader, test_loader = make_loaders(args.metadata, args.feature_type, args.batch_size)
    model = build_model(args.model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)[0] if isinstance(model, BiLSTMEmotionNet) else model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            loop.set_postfix(loss=loss.item())

        train_loss = epoch_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = ckpt_dir / f"best_{args.model}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "model_type": args.model,
                "feature_type": args.feature_type,
                "label_order": EMOTION_ORDER,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepEmotion models")
    parser.add_argument("--metadata", type=str, required=True, help="metadata_features.csv path")
    parser.add_argument("--model", type=str, choices=["cnn", "bilstm", "hybrid"], default="cnn")
    parser.add_argument("--feature_type", type=str, choices=["mel", "mfcc"], default="mel")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default="outputs/logs")
    parser.add_argument("--checkpoint_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)
