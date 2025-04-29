import pandas as pd
import os
import librosa
import numpy as np
from itertools import groupby
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# Load dataset
df = pd.read_csv("mfcc_dataset.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Label mapping
label_map = {0: "Whistle", 1: "Click", 2: "Burst Pulse"}

def class_counts(labels):
    counts = Counter(labels)
    msg = "-> "
    for k in sorted(counts):
        msg += f"{label_map.get(k, 'Unknown')}: {counts[k]}  "
    return msg.strip()

print(f"Loaded: {len(X_scaled)} || {class_counts(y)}")
print(f"X_train: {len(X_train)}, y_train: {len(y_train)} || {class_counts(y_train)}")
print(f"X_test: {len(X_test)}, y_test: {len(y_test)} || {class_counts(y_test)}")

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Whistles", "Clicks", "Burst-Pulses"]))

# Normalize audio volume
def rmsNormalize(audio, targetdBFS=-20):
    rms = np.sqrt(np.mean(audio**2))
    scaler = 10 ** (targetdBFS / 20) / (rms + 1e-6)
    return audio * scaler

# Extract features
def extract_features_from_window(window, sr):
    mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=39, n_fft=2048)
    mean_mfcc = np.mean(mfcc, axis=1)
    return mean_mfcc

# Classify long WAV files
def classify_long_wav(filepath, clf, scaler, window_size_sec=1.0, sr_target=16000):
    y, sr = librosa.load(filepath, sr=sr_target)
    y = rmsNormalize(y)

    samples_per_window = int(window_size_sec * sr_target)
    n_windows = len(y) // samples_per_window

    preds = []
    for i in range(n_windows):
        start = i * samples_per_window
        end = start + samples_per_window
        chunk = y[start:end]
        if len(chunk) < samples_per_window:
            continue  # Skip incomplete window

        # Extract MFCCs
        mean_mfcc = extract_features_from_window(chunk, sr_target).reshape(1, -1)

        # FIX: Wrap features into a DataFrame with correct column names
        mean_mfcc_df = pd.DataFrame(mean_mfcc, columns=X.columns)

        # Scale features
        mfcc_scaled = scaler.transform(mean_mfcc_df)

        # Predict
        pred = clf.predict(mfcc_scaled)[0]
        preds.append(pred)

    # Group consecutive predictions
    grouped_preds = [key for key, _ in groupby(preds)]

    # Count each class in grouped preds
    count = Counter(grouped_preds)

    return preds, grouped_preds, count

# Classify tracks
TRACK_FILES = os.path.join('Data', '_Tracks')
for file in os.listdir(TRACK_FILES):
    filepath = os.path.join(TRACK_FILES, file)
    if os.path.isfile(filepath):
        yhat, grouped_preds, count = classify_long_wav(filepath, clf, scaler)

        if grouped_preds:
            ordered_counts = []
            seen = set()
            for label in grouped_preds:
                if label not in seen:
                    ordered_counts.append((label, count[label]))
                    seen.add(label)

            result = ', '.join(
                f"{num} {label_map.get(label, 'Unknown')}{' detected' if num == 1 else 's detected'}"
                for label, num in ordered_counts
            )
        else:
            result = "No sounds detected"

        print(f"Track: {file} || {result}")


"""
Loaded: 380 || -> Whistle: 114  Click: 220  Burst Pulse: 46
X_train: 304, y_train: 304 || -> Whistle: 83  Click: 183  Burst Pulse: 38
X_test: 76, y_test: 76 || -> Whistle: 31  Click: 37  Burst Pulse: 8

[[20  0  1]
 [ 0 40  0]
[[20  0  1]
 [ 0 40  0]
 [ 1  0  8]]
 [ 1  0  8]]
              precision    recall  f1-score   support
              precision    recall  f1-score   support


    Whistles       0.95      0.95      0.95        21
      Clicks       1.00      1.00      1.00        40
Burst-Pulses       0.89      0.89      0.89         9
Burst-Pulses       0.89      0.89      0.89         9

    accuracy                           0.97        70

    accuracy                           0.97        70
   macro avg       0.95      0.95      0.95        70
weighted avg       0.97      0.97      0.97        70

Track: TrackTest1.wav || 1 Burst Pulse detected
Track: TrackTest2.wav || 1 Burst Pulse detected
Track: TrackTest3.wav || No sounds detected
Track: TrackTest4.wav || No sounds detected
Track: TrackTest5.wav || 1 Burst Pulse detected
"""
