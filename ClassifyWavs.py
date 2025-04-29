import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import groupby
from collections import Counter
import joblib

clf = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.joblib")

label_map = {0: "Whistle", 1: "Click", 2: "Burst Pulse"}

def rmsNormalize(audio, targetdBFS=-20):
    rms = np.sqrt(np.mean(audio**2))
    scaler = 10 ** (targetdBFS / 20) / (rms + 1e-6)
    return audio * scaler

def extract_features_from_window(window, sr):
    mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=39, n_fft=2048)
    mean_mfcc = np.mean(mfcc, axis=1)
    return mean_mfcc

def classify_long_wav(file_path, window_size_sec=1.0, hop_size_sec=1.0):
    y, sr = librosa.load(file_path, sr=None)
    y = rmsNormalize(y)

    window_length = int(window_size_sec * sr)
    hop_length = int(hop_size_sec * sr)

    features = []
    for start in range(0, len(y) - window_length + 1, hop_length):
        window = y[start:start + window_length]
        feats = extract_features_from_window(window, sr)
        features.append(feats)

    X = scaler.transform(features)
    yhat = clf.predict(X)

    # Optionally reduce repetitions (group consecutive same values)
    grouped = [key for key, _ in groupby(yhat)]

    # Count final predictions
    count = Counter(grouped)
    print(f"Results for {os.path.basename(file_path)}:")
    for label, c in count.items():
        print(f"  {label_map[label]}: {c}")

    return yhat, grouped, count

# === EXAMPLE USAGE ===
yhat, grouped_preds, count = classify_long_wav("Data\Tracks\TrackTest1.wav")