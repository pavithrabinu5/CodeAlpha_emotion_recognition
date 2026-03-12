"""Hybrid CNN + BiLSTM model."""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNBiLSTMEmotionNet(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.cnn = CNNEncoder()
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, mel_bins, time)
        feat = self.cnn(x)  # (b, 64, m', t')
        b, c, m, t = feat.shape
        feat = feat.permute(0, 3, 1, 2).contiguous()  # (b, t', c, m')
        feat = feat.view(b, t, c * m)  # flatten frequency dims
        outputs, _ = self.lstm(feat)
        pooled = outputs.mean(dim=1)
        return self.fc(pooled)
