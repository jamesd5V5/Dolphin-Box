import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import tensorflow as tf
import torch.nn.functional as F
from CNN import CNN
from MultiLabelCNN import MultiLabelCNN
from DataExtraction import loadMono, loadLogMelSpectrogram, split_audio_tf, min_samples, apply_pca, extract_mfcc
import numpy as np
import joblib
from collections import defaultdict
from scipy.signal import butter, lfilter

sample_rate = 48000

# Temperature values to experiment with
TEMPERATURES = [0.5, 1.0, 1.5, 2.0]
input_size = 50 + 7 # 50 PCA + 7 MFCC
model = MultiLabelCNN(input_size=input_size)
model.load_state_dict(torch.load('best_multilabel_classifier.pt'))
model.eval()

class_names = ['Whistles', 'Clicks', 'BPs', 'Noise']
THRESHOLD = 0.55
MIN_CONFIDENCE_DIFF = 0.15

pca = joblib.load('PCAObject.joblib')

def highpass_filter(wav, sr, cutoff=1000, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_wav = lfilter(b, a, wav)
    return filtered_wav

def classify_wav(filepath, temperature):
    wav = loadMono(filepath)
    wav = highpass_filter(wav.numpy(), sample_rate, cutoff=1000)
    wav = tf.convert_to_tensor(wav, dtype=tf.float32)
    segments = split_audio_tf(wav, min_samples)
    
    predictions = []
    all_probs = defaultdict(list)
    segment_probs = []  # Store probabilities for each segment
    
    for index, segment in enumerate(segments):
        spec, _ = loadLogMelSpectrogram(segment, label=0)
        spec_np = spec.numpy()
        
        # Get PCA features
        reduced_spec = pca.transform(spec_np.reshape(1, -1))
        
        # Get MFCC features
        mfcc_features = extract_mfcc(segment, n_mfcc=13, n_keep=7)
        
        # Concatenate PCA and MFCC features
        combined_features = np.concatenate([reduced_spec, mfcc_features.reshape(1, -1)], axis=1)
        
        input_tensor = torch.tensor(combined_features, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
            # Handle the output properly - ensure it's a tensor
            if isinstance(output, tuple):
                output = output[0]  # Take the first element if it's a tuple
            # Apply temperature scaling
            scaled_output = output / temperature
            probs = torch.sigmoid(scaled_output)
            
            prob_values = probs[0].cpu().numpy()
            sorted_indices = np.argsort(prob_values)[::-1]
            
            # Store probabilities for analysis
            for i, class_name in enumerate(class_names):
                all_probs[class_name].append(prob_values[i])
            
            # Store probabilities for this segment
            segment_prob = {
                "Whistle": round(float(prob_values[0]), 2),  # Whistles
                "Click": round(float(prob_values[1]), 2),    # Clicks
                "BP": round(float(prob_values[2]), 2)        # BPs
            }
            segment_probs.append(segment_prob)
            
            if prob_values[sorted_indices[0]] - prob_values[sorted_indices[1]] >= MIN_CONFIDENCE_DIFF:
                present_classes = [class_names[i] for i, p in enumerate(prob_values) 
                                 if p >= THRESHOLD and p == prob_values[sorted_indices[0]]]
            else:
                present_classes = [class_names[i] for i, p in enumerate(prob_values) 
                                 if p >= THRESHOLD]
            
            print(f"W:{index+1} CLs:", ", ".join(f"{class_name}: {prob_values[i]:.2f}" 
                  for i, class_name in enumerate(class_names)))

            if present_classes:
                class_probs = [f"{prob_values[i]:.2f}" for i in range(len(class_names)) 
                             if prob_values[i] >= THRESHOLD]
                predictions.append(f"{' & '.join(present_classes)} ({', '.join(class_probs)})")
            else:
                predictions.append(f"Unknown (all below {THRESHOLD})")
    
    # Calculate average probabilities for each class
    avg_probs = {class_name: np.mean(probs) for class_name, probs in all_probs.items()}
    
    # Save segment probabilities to JSON
    import json
    with open('results.json', 'w') as f:
        json.dump(segment_probs, f)
    
    return predictions, avg_probs

def run_temperature_experiment():
    TRACK_FILES = os.path.join('Data', '_Tracks')
    results = {}
    
    for temperature in TEMPERATURES:
        print(f"\n{'='*50}")
        print(f"Testing with Temperature = {temperature}")
        print(f"{'='*50}")
        
        file_results = {}
        for filename in os.listdir(TRACK_FILES):
            if filename.endswith('.wav'):
                filepath = os.path.join(TRACK_FILES, filename)
                print(f"\n{filename}==================================")
                preds, avg_probs = classify_wav(filepath, temperature)
                print(f"Predictions: {preds}")
                print(f"Average probabilities: {avg_probs}")
                file_results[filename] = {
                    'predictions': preds,
                    'average_probabilities': avg_probs
                }
        
        results[temperature] = file_results
    
    return results

if __name__ == "__main__":
    results = run_temperature_experiment()
    
    # Print summary of results
    print("\n\nTemperature Experiment Summary")
    print("="*50)
    for temperature in TEMPERATURES:
        print(f"\nTemperature: {temperature}")
        print("-"*30)
        for filename, result in results[temperature].items():
            print(f"\nFile: {filename}")
            print(f"Average probabilities:")
            for class_name, prob in result['average_probabilities'].items():
                print(f"  {class_name}: {prob:.3f}")
