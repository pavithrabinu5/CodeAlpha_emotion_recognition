"""BiLSTM with attention for sequential MFCC features."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, time, hidden)
        weights = F.softmax(self.proj(x), dim=1)  # (batch, time, 1)
        context = (weights * x).sum(dim=1)  # (batch, hidden)
        return context, weights.squeeze(-1)


class BiLSTMEmotionNet(nn.Module):
    def __init__(self, input_dim: int = 40 * 2, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 7, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, time, features)
        outputs, _ = self.lstm(x)
        context, attn_weights = self.attention(outputs)
        logits = self.fc(context)
        return logits, attn_weights
