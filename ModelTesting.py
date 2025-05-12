import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import tensorflow as tf
import torch.nn.functional as F
from CNN import CNN
from DataExtraction import loadMono, loadLogMelSpectrogram, split_audio_tf, min_samples, apply_pca
import numpy as np
import joblib

model = CNN(input_size=50)
model.load_state_dict(torch.load('dolp_classifier.pt'))
model.eval()

class_names = ['Whistles', "Clicks", "BPs"]
THRESHOLD = 0.6

pca = joblib.load('PCAObject.joblib')

def classify_wav(filepath):
    wav = loadMono(filepath)
    segments = split_audio_tf(wav, min_samples)
    
    predictions = []
    
    for index, segment in enumerate(segments):
        spec, _ = loadLogMelSpectrogram(segment, label=0)
        spec_np = spec.numpy()
        
        reduced_spec = pca.transform(spec_np.reshape(1, -1))
        input_tensor = torch.tensor(reduced_spec, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.sigmoid(output)
            present_classes = [class_names[i] for i, p in enumerate(probs[0]) if p.item() >= THRESHOLD]
            print(f"W:{index+1} CLs:", ", ".join(f"{class_name}: {probs[0][i].item():.2f}" for i, class_name in enumerate(class_names)))

            if present_classes:
                predictions.append(f"{' & '.join(present_classes)} ({', '.join(f'{probs[0][i].item():.2f}' for i in range(len(class_names)) if probs[0][i].item() >= THRESHOLD)})")
            else:
                predictions.append(f"Unknown (all below {THRESHOLD})")
    
    return predictions

TRACK_FILES = os.path.join('Data', '_Tracks')
for filename in os.listdir(TRACK_FILES):
    if filename.endswith('.wav'):
        filepath = os.path.join(TRACK_FILES, filename)
        print(f"{filename}==================================")
        preds = classify_wav(filepath)
        print(f"{filename}: {preds}")
