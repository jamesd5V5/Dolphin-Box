import os
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import librosa
import random
import scipy.ndimage
from sklearn.decomposition import PCA
from collections import defaultdict
import joblib

WHISTLE_FILES = os.path.join('Data', 'Whistles')
CLICK_FILES = os.path.join('Data', 'Clicks')
BP_FILES = os.path.join('Data', 'BPs')
NOISE_FILES = os.path.join('Data', 'Noise')

n_fft = 2048  #1024
hop_length = 512  #256
n_mels = 200
sample_rate = 48000
cutoff_freq = sample_rate // 2 #36000  
duration = 1 #seconds
min_samples = int(duration * sample_rate)
noise_threshold_db = -2 #-2 works best so far

n_components = 50  # Using 50 components for better accuracy

TEMPERATURE = 1.0  # More confident predictions

def loadMono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    return wav

def loadLogMelSpectrogram(wav, label):
    wav = wav[:min_samples]
    padding = tf.zeros([min_samples - tf.shape(wav)[0]], dtype=tf.float32)
    wav = tf.concat([wav, padding], 0)

    stft = tf.signal.stft(wav, frame_length=n_fft, frame_step=hop_length)
    magnitude = tf.abs(stft)

    num_spectrogram_bins = (n_fft // 2) + 1
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=0,
        upper_edge_hertz=cutoff_freq
    )

    mel_spectrogram = tf.matmul(magnitude, mel_weights)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    noise_threshold = tf.constant(noise_threshold_db, dtype=tf.float32)
    log_mel_spectrogram = tf.where(log_mel_spectrogram < noise_threshold, noise_threshold, log_mel_spectrogram)

    log_mel_spectrogram = normalize(log_mel_spectrogram, method="minmax")
    log_mel_spectrogram = remove_small_blobs(log_mel_spectrogram)

    return log_mel_spectrogram, label

def split_audio_tf(audio, target_length, min_valid_length=None):
    if min_valid_length is None:
        min_valid_length = target_length // 2

    segments = []
    total_length = tf.shape(audio)[0]
    num_segments = total_length // target_length

    for i in range(num_segments):
        segment = audio[i * target_length : (i + 1) * target_length]
        segments.append(segment)

    leftover = total_length % target_length
    if leftover >= min_valid_length:
        last_segment = audio[-leftover:]
        padding = tf.zeros([target_length - leftover], dtype=tf.float32)
        last_segment = tf.concat([last_segment, padding], 0)
        segments.append(last_segment)

    return segments

#morphological filtering/ CCA (Connected Componnent analysis)
def remove_small_blobs(spec, threshold=0.3, min_size=10):
    binary_spec = (spec > threshold).numpy().astype(np.uint8)
    labeled_array, num_features = scipy.ndimage.label(binary_spec)

    sizes = np.bincount(labeled_array.ravel())

    mask_sizes = sizes >= min_size
    mask_sizes[0] = 0 
    cleaned = mask_sizes[labeled_array]
    
    cleaned_spec = spec * tf.convert_to_tensor(cleaned, dtype=tf.float32)
    return cleaned_spec

def loadDataset(directory, label):
    dataset = []
    for fname in os.listdir(directory):
        if fname.endswith('.wav'):
            path = os.path.join(directory, fname)
            wav = loadMono(path)

            segments = split_audio_tf(wav, min_samples)

            for segment in segments:
                spec, lbl = loadLogMelSpectrogram(segment, label)
                dataset.append((spec.numpy(), lbl))

    return dataset

def normalize(spec, method="minmax"):
    if method == "minmax":
        return (spec - tf.reduce_min(spec)) / (tf.reduce_max(spec) - tf.reduce_min(spec) + 1e-6)
    elif method == "standard":
        return (spec - tf.reduce_mean(spec)) / (tf.math.reduce_std(spec) + 1e-6)
    else:
        return spec

def plot_all_spectrograms(dataset, rows=6, cols=5):
    label_map = {0: "Whistle", 1: "Click", 2: "Burst Pulse", 3: "Noise"}
    label_counts = defaultdict(int)

    plt.figure(figsize=(cols * 5, rows * 3))
    for i, (spec, label) in enumerate(dataset[:cols * rows]):
        label_counts[label] += 1
        title = f"{label_map.get(label, 'Unknown')} #{label_counts[label]}"
        
        plt.subplot(rows, cols, i + 1)
        plt.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
        plt.title(title)
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("spectrogram_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved spectrogram grid to spectrogram_grid.png")

def get_each_spectrogram():
    whistle_data = loadDataset(WHISTLE_FILES, label=0)
    click_data = loadDataset(CLICK_FILES, label=1)
    bp_data = loadDataset(BP_FILES, label=2)
    noise_data = loadDataset(NOISE_FILES, label=3)
    data = whistle_data + click_data + bp_data + noise_data
    print(f"Loaded {len(whistle_data)} Whistles, {len(click_data)} Clicks, {len(bp_data)} Bps, {len(noise_data)} Noise")
    return whistle_data, click_data, bp_data, noise_data

def get_all_spectrograms(cache_file='SpectrogramCache.npz'):
    if os.path.exists(cache_file):
        print(f"Loading spectrograms from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        specs = data['spectrograms']
        labels = data['labels']
        return list(zip(specs, labels))
    else:
        whistle_data, click_data, bp_data, noise_data = get_each_spectrogram()
        all_data = whistle_data + click_data + bp_data + noise_data
        specs = [spec for spec, _ in all_data]
        labels = [label for _, label in all_data]
        np.savez_compressed(cache_file, spectrograms=specs, labels=labels)
        print(f"Saved processed spectrograms to {cache_file}")
        return all_data

def apply_pca(specs, n_components=n_components):
    flattened_specs = np.array([spec.flatten() for spec in specs])
    
    pca = PCA(n_components=n_components)
    reduced_specs = pca.fit_transform(flattened_specs)
    
    return reduced_specs, pca

def extract_mfcc(segment, sample_rate=48000, n_mfcc=13, n_keep=7):
    if hasattr(segment, 'numpy'):
        segment = segment.numpy()
    segment = np.array(segment, dtype=np.float32)
    mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)  # shape (n_mfcc,)
    mfccs_mean = np.squeeze(mfccs_mean)
    mfccs_mean = mfccs_mean[:n_keep]  # Keep only the first n_keep coefficients
    return mfccs_mean

def get_all_spectrograms_with_pca(cache_file='SpectrogramCache.npz', pca_cache_file='PCACache.npz', n_mfcc=13, n_keep=7):
    if os.path.exists(pca_cache_file):
        print(f"Loading PCA components from {pca_cache_file}")
        data = np.load(pca_cache_file, allow_pickle=True)
        reduced_specs = data['reduced_specs']
        labels = data['labels']
        # Load MFCCs if present
        if 'mfccs' in data:
            mfccs = data['mfccs']
            features = [np.concatenate([pca_feat, np.squeeze(mfcc_feat)[:n_keep]]) for pca_feat, mfcc_feat in zip(reduced_specs, mfccs)]
            return list(zip(features, labels))
        else:
            return list(zip(reduced_specs, labels))
    else:
        all_data = get_all_spectrograms(cache_file)
        specs = [spec for spec, _ in all_data]
        labels = [label for _, label in all_data]
        reduced_specs, pca = apply_pca(specs, n_components)
        mfccs = [extract_mfcc(spec, n_mfcc=n_mfcc, n_keep=n_keep) for spec in specs]
        np.savez_compressed(pca_cache_file, reduced_specs=reduced_specs, labels=labels, mfccs=mfccs)
        joblib.dump(pca, 'PCAObject.joblib')
        print(f"Saved PCA components and MFCCs to {pca_cache_file} and PCA object to PCAObject.joblib")
        features = [np.concatenate([pca_feat, np.squeeze(mfcc_feat)]) for pca_feat, mfcc_feat in zip(reduced_specs, mfccs)]
        return list(zip(features, labels))

def plot_random_spectrograms(sizePerClass=10):
    whistle_data, click_data, bp_data, noise_data = get_each_spectrogram()
    whistle_samples = random.sample(whistle_data, min(sizePerClass, len(whistle_data)))
    click_samples = random.sample(click_data, min(sizePerClass, len(click_data)))
    bp_samples = random.sample(bp_data, min(sizePerClass, len(bp_data)))
    noise_samples = random.sample(noise_data, min(sizePerClass, len(noise_data)))

    samples_to_plot = whistle_samples + click_samples + bp_samples + noise_samples
    plot_all_spectrograms(samples_to_plot)

#plot_random_spectrograms(10)