# Emotion Recognition from Speech

A deep learning pipeline for classifying human emotions from raw audio using a hybrid CNN-BiLSTM architecture with attention mechanisms. Achieves **86% test accuracy** across 8 emotion classes on the RAVDESS and TESS benchmark datasets.

---

## Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 86.01% |
| Validation Accuracy | 82.55% |
| Macro F1-Score | 0.85 |
| Train/Val Gap | ~6.5% |
| Training Epochs | 69 (early stop) |

### Per-Class Results

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 0.88 | 0.89 | 0.88 |
| Calm | 0.53 | 0.90 | 0.67 |
| Disgust | 0.85 | 0.85 | 0.85 |
| Fearful | 0.91 | 0.83 | 0.87 |
| Happy | 0.90 | 0.79 | 0.84 |
| Neutral | 0.91 | 0.92 | 0.91 |
| Sad | 0.92 | 0.78 | 0.84 |
| Surprised | 0.86 | 0.97 | 0.91 |

---

## Architecture

```
Input Audio (.wav)
      │
      ▼
Feature Extraction
  ├── MFCC (40 coefficients + delta + delta-delta)
  ├── Mel Spectrogram (40 bands)
  └── Chroma STFT (8 bins)
      │  → stacked → (128, 128) feature map
      ▼
CNN Block 1: Conv2d(1→32) → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
      │
CNN Block 2: Conv2d(32→64) → BN → ReLU → Conv2d → BN → ReLU → MaxPool → Dropout
      │
CNN Block 3: Conv2d(64→128) → BN → ReLU → MaxPool → Dropout
      │
Squeeze-and-Excitation Block (channel-wise attention)
      │
Reshape → (batch, 16, 2048)
      │
Bidirectional LSTM (128 units × 2 layers, dropout=0.3)
      │
Self-Attention (temporal context pooling)
      │
Dense(256) → BN → ReLU → Dropout(0.4)
      │
Dense(num_classes) → Softmax
```

**Total Parameters:** ~2.8M

---

## Datasets

| Dataset | Samples | Emotions | Speakers |
|---------|---------|----------|----------|
| RAVDESS | 1,440 | 8 | 24 actors |
| TESS | 2,800 | 7 | 2 actresses |
| **Combined** | **4,240** | **8** | **26** |

Emotion label mapping is unified across both datasets (e.g., TESS `fear` → `fearful`, `ps` → `surprised`).

---

## Project Structure

```
emotion_recognition/
├── src/
│   ├── data_loader.py     # RAVDESS and TESS dataset parsers
│   ├── features.py        # Feature extraction and dataset preprocessing
│   ├── model.py           # CNN-BiLSTM-Attention model definition
│   ├── train.py           # Training loop with early stopping and LR scheduling
│   └── predict.py         # Single-file inference with confidence scores
├── data/
│   ├── raw/               # Raw audio datasets
│   └── processed/         # Cached .npy feature arrays
├── models/
│   ├── best_model.pt          # Best checkpoint (by val accuracy)
│   ├── training_history.png   # Accuracy and loss curves
│   └── confusion_matrix.png   # Per-class prediction breakdown
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+, macOS (Apple Silicon) or Linux

```bash
git clone https://github.com/yourname/emotion_recognition
cd emotion_recognition
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install librosa numpy pandas scikit-learn matplotlib seaborn tqdm soundfile jupyter
```

---

## Usage

### Inference on a single audio file

```bash
python3 src/predict.py /path/to/audio.wav
```

Output:
```
File: /path/to/audio.wav
-----------------------------------
Predicted Emotion: ANGRY
-----------------------------------
Top 3 probabilities:
  angry         78.5%  ████████████████████████
  happy          8.8%  ██
  disgust        6.6%  █
```

### Reproduce training from scratch

```bash
# Step 1: Extract features from raw audio
python3 src/features.py

# Step 2: Train model
python3 src/train.py
```

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 5e-3 |
| Batch Size | 32 |
| Max Epochs | 100 |
| Early Stop Patience | 20 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=7) |
| Label Smoothing | 0.1 |
| Gradient Clipping | 0.5 |

### Data Augmentation (training only)

- SpecAugment-style time masking (width 10–30 frames)
- Frequency masking (width 5–20 bins)
- Gaussian noise (σ = 0.015)
- Random gain scaling (0.85–1.15)

---

## Feature Engineering

Each audio file is processed as follows:

1. Resampled to 22,050 Hz, trimmed/padded to 3 seconds
2. Pre-emphasis filter applied (α = 0.97)
3. Features extracted and stacked into a (128, 128) 2D array:
   - 40 MFCCs
   - 40 delta MFCCs (velocity)
   - 40 delta-delta MFCCs (acceleration)
   - 40 Mel spectrogram bands (log scale)
   - 8 chroma features
4. Per-row (per-feature) z-score normalization

---

## Hardware

Trained on Apple M4 (MPS backend via PyTorch). Training time approximately 25 minutes for 69 epochs on 4,240 samples.

---
