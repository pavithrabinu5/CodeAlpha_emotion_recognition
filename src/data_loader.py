import os
import pandas as pd

RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm',    '03': 'happy',   '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust',  '08': 'surprised'
}

def load_ravdess(data_dir):
    records = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith('.wav'):
                parts = fname.split('-')
                emotion_code = parts[2]
                emotion = RAVDESS_EMOTIONS.get(emotion_code)
                if emotion:
                    records.append({
                        'path': os.path.join(root, fname),
                        'emotion': emotion
                    })
    df = pd.DataFrame(records)
    print(f"✅ Loaded {len(df)} audio files")
    print(df['emotion'].value_counts())
    return df

if __name__ == "__main__":
    df = load_ravdess("/Users/pavithrabinu/Downloads/Audio_Speech_Actors_01-24")
    print(df.head())