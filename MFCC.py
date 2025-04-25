import os
import librosa
import numpy as np
import csv

# Paths
WHISTLE_FILES = os.path.join('Data', 'Whistles')
CLICK_FILES = os.path.join('Data', 'Clicks')
BP_FILES = os.path.join('Data', 'BPs')

label_map = {0: "Whistle", 1: "Click", 2: "Burst Pulse"}

def loadDataset(directory, label):
    dataset = []
    for fname in os.listdir(directory):
        if fname.endswith('.wav'):
            path = os.path.join(directory, fname)
            try:
                y, sr = librosa.load(path, sr=None)
                if len(y) == 0:
                    print(f"Warning: {fname} is empty, skipping.")
                    continue
                n_fft = 512 if len(y) < 2048 else 2048
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
                mean_mfcc = np.mean(mfcc, axis=1)
                dataset.append((mean_mfcc, label))

            except Exception as e:
                print(f"Error processing {fname}: {e}")
    return dataset

whistle_data = loadDataset(WHISTLE_FILES, 0)
click_data = loadDataset(CLICK_FILES, 1)
bp_data = loadDataset(BP_FILES, 2)

print(f"Loaded {len(whistle_data)} Whistles, {len(click_data)} Clicks, {len(bp_data)} Bps")

full_dataset = whistle_data + click_data + bp_data

X = np.array([entry[0] for entry in full_dataset])  # Features
y = np.array([entry[1] for entry in full_dataset])  # Labels

output_csv = 'mfcc_dataset.csv'
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = [f'mfcc_{i+1}' for i in range(13)] + ['label']
    writer.writerow(header)

    for features, label in full_dataset:
        writer.writerow(list(features) + [label])

def computeClassStats(dataset, num_features=13):
    class_features = {}
    for features, label in dataset:
        if label not in class_features:
            class_features[label] = []
        class_features[label].append(features)
    class_stats = {}
    for label, feature_list in class_features.items():
        stacked = np.vstack(feature_list)
        mean_vector = np.mean(stacked, axis=0)
        std_vector = np.std(stacked, axis=0)
        class_stats[label] = (mean_vector, std_vector)

    return class_stats

stats = computeClassStats(full_dataset)

stats_csv = 'mfcc_class_stats.csv'
with open(stats_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = [f'mean_mfcc_{i+1}' for i in range(13)] + \
             [f'std_mfcc_{i+1}' for i in range(13)] + ['label']
    writer.writerow(header)

    for label, (mean_vec, std_vec) in stats.items():
        writer.writerow(list(mean_vec) + list(std_vec) + [label_map[label]])

for label, (mean_vec, std_vec) in stats.items():
    print(f"\n{label_map[label]} MFCC Features (sorted by std dev):")
    sorted = np.argsort(std_vec)
    for idx in sorted:
        print(f"MFCC {idx+1:>2}: Mean = {mean_vec[idx]:.4f}, Std = {std_vec[idx]:.4f}")


