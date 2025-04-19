import os
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import librosa

# Paths
WHISTLE_FILES = os.path.join('Data', 'Whistles')
CLICK_FILES = os.path.join('Data', 'Clicks')

# Audio processing params
n_fft = 2048
hop_length = 512
n_mels = 200
sample_rate = 48000
cutoff_freq = sample_rate // 2 #36000  
duration = 1 #seconds
min_samples = int(duration * sample_rate)

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
    return log_mel_spectrogram, label

def loadDataset(directory, label):
    dataset = []
    for fname in os.listdir(directory):
        if fname.endswith('.wav'):
            path = os.path.join(directory, fname)
            wav = loadMono(path)
            spec, lbl = loadLogMelSpectrogram(wav, label)
            dataset.append((spec.numpy(), lbl))
    return dataset

def plot_all_spectrograms(dataset, cols=4):
    rows = (len(dataset) + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows * 3))
    for i, (spec, label) in enumerate(dataset[:cols * rows]):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')
        plt.title("Whistle" if label == 1 else "Click")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("spectrogram_grid.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved spectrogram grid to spectrogram_grid.png")

def get_all_spectrograms():
    whistle_data = loadDataset(WHISTLE_FILES, label=1)
    click_data = loadDataset(CLICK_FILES, label=0)
    data = whistle_data + click_data
    print(f"Loaded {len(whistle_data)} Whistles, {len(click_data)} Clicks.")
    return data

plot_all_spectrograms(get_all_spectrograms())