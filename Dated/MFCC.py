import os
import librosa
import numpy as np
import csv
import librosa.display
import matplotlib.pyplot as plt
import random


# Paths
WHISTLE_FILES = os.path.join('Data', 'Whistles')
CLICK_FILES = os.path.join('Data', 'Clicks')
BP_FILES = os.path.join('Data', 'BPs')

label_map = {0: "Whistle", 1: "Click", 2: "Burst Pulse"}

def rmsNormalize(audio, targetdBFS=-20):
    rms = np.sqrt(np.mean(audio**2))
    scaler = 10 ** (targetdBFS / 20) / (rms + 1e-6)
    return audio * scaler

def pad_or_trim_audio(audio, target_length):
    # Pad the audio to the target length
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
    # If the audio is longer, split it into chunks of target length
    elif len(audio) > target_length:
        audio = audio[:target_length]  # trim audio to the target length for now
    return audio

def split_audio(audio, target_length):
    # Split longer audio into smaller segments
    if len(audio) > target_length:
        num_chunks = len(audio) // target_length
        segments = [audio[i*target_length:(i+1)*target_length] for i in range(num_chunks)]
        if len(audio) % target_length != 0:  # Handle leftover part
            segments.append(audio[num_chunks*target_length:])
        return segments
    else:
        return [audio]  # Return the original audio if it's already shorter than target_length

def loadDataset(directory, label, target_length=16000):
    dataset = []
    for fname in os.listdir(directory):
        if fname.endswith('.wav'):
            path = os.path.join(directory, fname)
            try:
                y, sr = librosa.load(path, sr=None)
                y = rmsNormalize(y, targetdBFS=-20)
                if len(y) == 0:
                    print(f"Warning: {fname} is empty, skipping.")
                    continue
                
                audio_segments = split_audio(y, target_length)

                for segment in audio_segments:
                    segment = pad_or_trim_audio(segment, target_length)
                    n_fft = 512 if len(segment) < 2048 else 2048
                    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=39, n_fft=n_fft)
                    mean_mfcc = np.mean(mfcc, axis=1)
                    dataset.append((mfcc, label, sr))  # keep sr for plotting later

            except Exception as e:
                print(f"Error processing {fname}: {e}")
    return dataset

# Set a target length for the audio samples (e.g., 1 second at 16kHz = 16000 samples)
target_length = 16000

whistle_data = loadDataset(WHISTLE_FILES, 0, target_length)
click_data = loadDataset(CLICK_FILES, 1, target_length)
bp_data = loadDataset(BP_FILES, 2, target_length)

print(f"Loaded {len(whistle_data)} Whistles, {len(click_data)} Clicks, {len(bp_data)} Bps")

full_dataset = whistle_data + click_data + bp_data

X = np.array([entry[0] for entry in full_dataset])  # Features
y = np.array([entry[1] for entry in full_dataset])  # Labels

output_csv = 'mfcc_dataset.csv'
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = [f'mfcc_{i+1}' for i in range(39)] + ['label']
    writer.writerow(header)

    for features, label, sr in full_dataset:
        features = features.flatten().tolist()  # ensure it's a plain list of floats
        writer.writerow(features + [label])

def computeClassStats(dataset, num_features=39):
    class_features = {}
    for features, label, sr in dataset:
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
    header = [f'mean_mfcc_{i+1}' for i in range(39)] + \
             [f'std_mfcc_{i+1}' for i in range(39)] + ['label']
    writer.writerow(header)

    for label, (mean_vec, std_vec) in stats.items():
        mean_vec = mean_vec.flatten().tolist()
        std_vec = std_vec.flatten().tolist()
        writer.writerow(mean_vec + std_vec + [label_map[label]])

for label, (mean_vec, std_vec) in stats.items():
    print(f"\n{label_map[label]} MFCC Features (sorted by std dev):")
    sorted = np.argsort(std_vec)
    for idx in sorted:
        print(f"MFCC {idx+1:>2}: Mean = {mean_vec[idx]:.4f}, Std = {std_vec[idx]:.4f}")


picSize = 10
#Pics
whistle_samples = random.sample(whistle_data, min(picSize, len(whistle_data)))
click_samples = random.sample(click_data, min(picSize, len(click_data)))
bp_samples = random.sample(bp_data, min(picSize, len(bp_data)))

# Combine them
samples_to_plot = whistle_samples + click_samples + bp_samples
print(f"Total samples to plot: {len(samples_to_plot)}")

num_images = len(samples_to_plot)
cols = 10  # e.g., 10 columns
rows = (num_images + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2))
axes = axes.flatten()

for i, (mfcc, label, sr) in enumerate(samples_to_plot):
    axes[i].axis('off')
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=axes[i])
    axes[i].set_title(label_map[label], fontsize=6)

# Hide extra empty plots
for i in range(len(samples_to_plot), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('sampled_mfcc_grid.png', dpi=300)
plt.close()