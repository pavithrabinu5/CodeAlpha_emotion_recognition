import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import EmotionModel

class EmotionDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Time mask
            if torch.rand(1) > 0.4:
                t = torch.randint(10, 30, (1,)).item()
                t0 = torch.randint(0, 128-t, (1,)).item()
                x[:, :, t0:t0+t] = 0
            # Freq mask
            if torch.rand(1) > 0.4:
                f = torch.randint(5, 20, (1,)).item()
                f0 = torch.randint(0, 128-f, (1,)).item()
                x[:, f0:f0+f, :] = 0
            # Double mask (SpecAugment style)
            if torch.rand(1) > 0.5:
                t = torch.randint(5, 15, (1,)).item()
                t0 = torch.randint(0, 128-t, (1,)).item()
                x[:, :, t0:t0+t] = 0
            # Noise
            if torch.rand(1) > 0.4:
                x += torch.randn_like(x) * 0.015
            # Random gain
            if torch.rand(1) > 0.5:
                x *= torch.FloatTensor(1).uniform_(0.85, 1.15)
        return x, self.y[idx]

def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train: optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            loss_sum += loss.item()
            correct += (out.argmax(1)==yb).sum().item()
            total += len(yb)
    return loss_sum/len(loader), correct/total

def main():
    BATCH   = 32
    EPOCHS  = 100
    LR      = 1e-4        # lower = less overfitting
    PAT     = 20
    DEVICE  = torch.device("mps")
    os.makedirs("models", exist_ok=True)

    print("📂 Loading data...")
    X = np.load("data/processed/X.npy").transpose(0,3,1,2)
    y = np.load("data/processed/y.npy")
    classes = np.load("data/processed/classes.npy", allow_pickle=True)
    print(f"✅ {len(X)} samples | {len(classes)} classes: {classes}")

    # 70/15/15 split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    print(f"Train:{len(X_tr)} | Val:{len(X_val)} | Test:{len(X_te)}")

    tr_dl  = DataLoader(EmotionDataset(X_tr,  y_tr,  augment=True),
                        batch_size=BATCH, shuffle=True,  num_workers=0)
    val_dl = DataLoader(EmotionDataset(X_val, y_val, augment=False),
                        batch_size=BATCH, shuffle=False, num_workers=0)
    te_dl  = DataLoader(EmotionDataset(X_te,  y_te,  augment=False),
                        batch_size=BATCH, shuffle=False, num_workers=0)

    model     = EmotionModel(len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7)

    best_acc, pat_cnt = 0, 0
    tr_accs, val_accs, tr_losses, val_losses = [], [], [], []

    print(f"\n🚀 Training on {DEVICE}...\n")
    for ep in range(1, EPOCHS+1):
        tl, ta = run_epoch(model, tr_dl,  optimizer, criterion, DEVICE, train=True)
        vl, va = run_epoch(model, val_dl, optimizer, criterion, DEVICE, train=False)
        scheduler.step(va)

        tr_accs.append(ta); val_accs.append(va)
        tr_losses.append(tl); val_losses.append(vl)

        gap = ta - va
        print(f"Epoch {ep:03d}/{EPOCHS} | "
              f"Train:{ta:.4f} Val:{va:.4f} Gap:{gap:.3f} | "
              f"LR:{optimizer.param_groups[0]['lr']:.6f}")

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"  💾 Saved! val_acc={va:.4f}")
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= PAT:
                print(f"\n⏹ Early stop at epoch {ep}")
                break

    # Test evaluation
    model.load_state_dict(torch.load("models/best_model.pt"))
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in te_dl:
            preds += model(xb.to(DEVICE)).argmax(1).cpu().tolist()
            trues += yb.tolist()

    print(f"\n🏆 Best val accuracy: {best_acc:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(trues, preds, target_names=classes))

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    ax[0].plot(tr_accs, label='Train', color='blue')
    ax[0].plot(val_accs, label='Val', color='orange')
    ax[0].set_title('Accuracy'); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(tr_losses, label='Train', color='blue')
    ax[1].plot(val_losses, label='Val', color='orange')
    ax[1].set_title('Loss'); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=150)

    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix — Test Acc: {sum(p==t for p,t in zip(preds,trues))/len(trues):.2%}')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150)
    plt.show()
    print("✅ Done! Check models/ for saved charts.")

if __name__ == "__main__":
    main()