import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import tensorflow as tf
import torch.nn.functional as F
from CNN import CNN
from DataExtraction import loadMono, loadLogMelSpectrogram, split_audio_tf, min_samples
import numpy as np


model = CNN()
model.load_state_dict(torch.load('dolp_classifier.pt'))
model.eval()

class_names = ['Whistles', "Clicks", "BPs"]
THRESHOLD = 0.6

def classify_wav(filepath):
    wav = loadMono(filepath)
    segments = split_audio_tf(wav, min_samples)

    predictions = []

    for index, segment in enumerate(segments):
        spec, _ = loadLogMelSpectrogram(segment, label=0)
        spec_np = spec.numpy()
        input = torch.tensor(spec_np).unsqueeze(0).unsqueeze(0).float() # [1,1,H,W]

        with torch.no_grad():
            output = model(input)
            probs = F.softmax(output, dim=1)
            topProb, predClass = torch.max(probs, dim=1)

            print(f"W:{index+1} CLs:", ", ".join(f"{class_name}: {probs[0][i].item():.2f}" for i, class_name in enumerate(class_names)))

                
            if topProb.item() >= THRESHOLD:
                predictions.append(f"{class_names[predClass.item()]} ({topProb.item():.2f}) with Lvl: {THRESHOLD}")
            else:
                predictions.append(f"Unknown ({topProb.item():.2f}) with Lvl: {THRESHOLD}")

    return predictions

TRACK_FILES = os.path.join('Data', '_Tracks')
for filename in os.listdir(TRACK_FILES):
    if filename.endswith('.wav'):
        filepath = os.path.join(TRACK_FILES, filename)
        print(f"{filename}==================================")
        preds = classify_wav(filepath)
        print(f"{filename}: {preds}")
