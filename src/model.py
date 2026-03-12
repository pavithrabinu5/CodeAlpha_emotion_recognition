import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1)

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//r), nn.ReLU(),
            nn.Linear(ch//r, ch), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)

class EmotionModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.se = SEBlock(128)
        # After 3x MaxPool: 128→16
        self.lstm = nn.LSTM(128*16, 128, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = SelfAttention(256)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.se(x)
        b, c, h, w = x.shape
        x = x.permute(0,2,1,3).reshape(b, h, c*w)
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.head(x)

# Single model — no ensemble (easier to train well)
if __name__ == "__main__":
    device = torch.device("mps")
    m = EmotionModel(8).to(device)
    dummy = torch.randn(4,1,128,128).to(device)
    print(f"✅ Output: {m(dummy).shape}")
    print(f"✅ Params: {sum(p.numel() for p in m.parameters()):,}")