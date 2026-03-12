import librosa
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pandas as pd

RAVDESS_DIR = "/Users/pavithrabinu/Downloads/Audio_Speech_Actors_01-24"
TESS_DIR    = "/Users/pavithrabinu/data/raw/TESS"

TESS_EMOTIONS = {
    'angry': 'angry', 'disgust': 'disgust', 'fear': 'fearful',
    'happy': 'happy', 'neutral': 'neutral', 'pleasant_surprise': 'surprised',
    'ps': 'surprised', 'sad': 'sad'
}

RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

def load_ravdess(data_dir):
    records = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith('.wav'):
                parts = fname.split('-')
                emotion = RAVDESS_EMOTIONS.get(parts[2])
                if emotion:
                    records.append({'path': os.path.join(root, fname), 'emotion': emotion})
    print(f"✅ RAVDESS: {len(records)} files")
    return records

def load_tess(data_dir):
    records = []
    if not os.path.exists(data_dir):
        print("⚠️  TESS not found")
        return records
    for folder in os.listdir(data_dir):
        fp = os.path.join(data_dir, folder)
        if not os.path.isdir(fp): continue
        fl = folder.lower()
        emotion = None
        for key, val in TESS_EMOTIONS.items():
            if key in fl:
                emotion = val
                break
        if emotion:
            for fname in os.listdir(fp):
                if fname.endswith('.wav'):
                    records.append({'path': os.path.join(fp, fname), 'emotion': emotion})
    print(f"✅ TESS: {len(records)} files")
    return records

def extract_features(file_path, sr=22050, duration=3):
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        target = sr * duration
        y = np.pad(y, (0, max(0, target - len(y))))[:target]
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])

        # Features
        mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_delta  = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mel         = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40), ref=np.max)
        chroma      = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=8)

        feat = np.vstack([mfcc, mfcc_delta, mfcc_delta2, mel, chroma])  # (128, T)

        # Normalize per feature row
        feat = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + 1e-9)

        # Fixed time = 128
        if feat.shape[1] < 128:
            feat = np.pad(feat, ((0,0),(0, 128 - feat.shape[1])))
        else:
            feat = feat[:, :128]

        return feat.astype(np.float32)
    except Exception as e:
        return None

def build_dataset(records, save_dir="data/processed"):
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(records)
    print(f"\n📊 Total: {len(df)}")
    print(df['emotion'].value_counts())

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['emotion'])

    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        feat = extract_features(row['path'])
        if feat is not None:
            X.append(feat)
            y.append(row['label'])

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    np.save(f"{save_dir}/X.npy", X)
    np.save(f"{save_dir}/y.npy", y)
    np.save(f"{save_dir}/classes.npy", le.classes_)
    print(f"✅ Saved {len(X)} samples shape: {X.shape}")
    return X, y, le.classes_

if __name__ == "__main__":
    records = load_ravdess(RAVDESS_DIR) + load_tess(TESS_DIR)
    print(f"🎯 Combined: {len(records)}")
    build_dataset(records)