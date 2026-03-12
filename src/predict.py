import torch
import librosa
import numpy as np
from model import EmotionModel

DEVICE  = torch.device("mps")
CLASSES = np.load("data/processed/classes.npy", allow_pickle=True)

def load_model():
    model = EmotionModel(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
    model.eval()
    return model

def extract_features(file_path, sr=22050, duration=3):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    target = sr * duration
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mel         = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), ref=np.max)
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=8)

    feat = np.vstack([mfcc, mfcc_delta, mfcc_delta2, mel, chroma])
    feat = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + 1e-9)
    if feat.shape[1] < 128:
        feat = np.pad(feat, ((0,0),(0,128-feat.shape[1])))
    else:
        feat = feat[:, :128]
    return feat.astype(np.float32)

def predict(file_path, model):
    feat = extract_features(file_path)
    x = torch.FloatTensor(feat).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).squeeze()

    top3 = probs.argsort(descending=True)[:3]

    print(f"\n🎤 File: {file_path}")
    print(f"{'─'*35}")
    print(f"🏆 Predicted Emotion: {CLASSES[probs.argmax()].upper()}")
    print(f"{'─'*35}")
    print("📊 Top 3 probabilities:")
    for i in top3:
        bar = '█' * int(probs[i].item() * 30)
        print(f"  {CLASSES[i]:12s} {probs[i].item()*100:5.1f}%  {bar}")
    return CLASSES[probs.argmax()]

if __name__ == "__main__":
    import sys
    model = load_model()
    if len(sys.argv) > 1:
        predict(sys.argv[1], model)
    else:
        # Test on a sample from the dataset
        import os
        test_file = None
        for root, _, files in os.walk("/Users/pavithrabinu/Downloads/Audio_Speech_Actors_01-24"):
            for f in files:
                if f.endswith('.wav'):
                    test_file = os.path.join(root, f)
                    break
            if test_file: break
        predict(test_file, model)