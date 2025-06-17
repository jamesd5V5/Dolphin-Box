import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DataExtraction import get_all_spectrograms_with_pca
from PrepData import PrepData
from MultiLabelCNN import MultiLabelCNN
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os

data = get_all_spectrograms_with_pca()
dataset = PrepData(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("\n=== Data Size Information ===")
print(f"Total number of samples: {len(data)}")
print(f"Input size (PCA + MFCC): {50 + 7}")
print(f"Batch size: {32}")
print(f"Number of batches per epoch: {len(dataloader)}")
print(f"Total samples per epoch: {len(dataloader) * 32}")

def convert_to_multi_hot(labels, num_classes=4):
    multi_hot = torch.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        multi_hot[i, label] = 1
    return multi_hot

labels = [label for _, label in data]
multi_hot_labels = convert_to_multi_hot(labels, num_classes=4)
print(f"Label shape: {multi_hot_labels.shape}")
class_counts = multi_hot_labels.sum(dim=0)
total_samples = len(labels)
print("\n=== Class Distribution ===")
for i, class_name in enumerate(['Whistle', 'Click', 'BP', 'Noise']):
    print(f"{class_name}: {class_counts[i]} samples ({class_counts[i]/total_samples*100:.1f}%)")

pos_weight = torch.tensor([2.0, 1.5, 2.0, 4.0])  # [Whistle, Click, BP, Noise]
print("Class weights:", pos_weight)
input_size = 50 + 7 # 50 PCA + 7 MFCC
model = MultiLabelCNN(input_size=input_size)

print("\n=== Model Architecture ===")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

class_criterion = nn.CrossEntropyLoss(weight=pos_weight)
conf_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
class_accuracies = {class_name: [] for class_name in ['Whistle', 'Click', 'BP', 'Noise']}
history = {
    'loss': [],
    'val_loss': [],
    'accuracy': [],
    'val_accuracy': []
}

num_epochs = 40
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        if batch_idx == 0 and epoch == 0:
            print("\n=== Batch Information ===")
            print(f"Input batch shape: {inputs.shape}")
            print(f"Labels batch shape: {labels.shape}")
        multi_hot = convert_to_multi_hot(labels)
        optimizer.zero_grad()
        class_output, conf_output = model(inputs)
        
        if batch_idx == 0 and epoch == 0:
            print(f"Classification output shape: {class_output.shape}")
            print(f"Confidence output shape: {conf_output.shape}")
        class_loss = class_criterion(class_output, labels)
        conf_loss = conf_criterion(conf_output, multi_hot)
        loss = class_loss + 0.5 * conf_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        running_loss += loss.item()
        with torch.no_grad():
            _, preds = torch.max(class_output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            confidences = torch.sigmoid(conf_output)
            all_confidences.extend(confidences.cpu().numpy())
    
    epoch_loss = running_loss/len(dataloader)
    print(f'\nEpoch {epoch+1}, Loss: {epoch_loss:.4f}')
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    if epoch == 0:
        print("\n=== Prediction Information ===")
        print(f"All predictions shape: {all_preds.shape}")
        print(f"All labels shape: {all_labels.shape}")
        print(f"All confidences shape: {all_confidences.shape}")
    epoch_accuracy = np.mean(all_preds == all_labels)
    history['loss'].append(epoch_loss)
    history['accuracy'].append(epoch_accuracy)
    
    output_line = ""
    for i, class_name in enumerate(['Whistle', 'Click', 'BP', 'Noise']):
        class_preds = all_preds == i
        class_labels = all_labels == i
        accuracy = (class_preds == class_labels).mean()
        precision = (class_preds & class_labels).sum() / (class_preds.sum() + 1e-6)
        recall = (class_preds & class_labels).sum() / (class_labels.sum() + 1e-6)
        avg_confidence = all_confidences[:, i].mean()
        output_line += f"{class_name}: [A({accuracy:.4f}), P({precision:.4f}), R({recall:.4f}), C({avg_confidence:.4f})], "
        class_accuracies[class_name].append(accuracy)
    print(output_line.rstrip(", "))

    scheduler.step(epoch_loss)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'multilabel_classifier.pt')

model.load_state_dict(torch.load('multilabel_classifier.pt'))
model.eval()
all_preds = []
all_labels = []
all_confidences = []
with torch.no_grad():
    for inputs, labels in dataloader:
        multi_hot = convert_to_multi_hot(labels)
        class_output, conf_output = model(inputs)
        _, preds = torch.max(class_output, 1)
        confidences = torch.sigmoid(conf_output)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_confidences = np.array(all_confidences)

with open('model_history.json', 'w') as f:
    json.dump(history, f)

np.save('y_true.npy', all_labels)
np.save('y_pred.npy', all_confidences)
test_data = inputs.cpu().numpy()
np.save('X_test.npy', test_data)

keras_model = Sequential([
    Dense(128, activation='relu', input_shape=(input_size,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(4, activation='sigmoid')
])

keras_model.save('multilabel_model.h5')

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Whistle', 'Click', 'BP', 'Noise']))
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
cm_overall = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Whistle', 'Click', 'BP', 'Noise'],
            yticklabels=['Whistle', 'Click', 'BP', 'Noise'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Overall Confusion Matrix')
plt.subplot(1, 2, 2)
epochs = range(1, num_epochs + 1)
for class_name in ['Whistle', 'Click', 'BP', 'Noise']:
    plt.plot(epochs, class_accuracies[class_name], label=class_name, marker='o')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Time for Each Class')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model_performance_metrics.png')
plt.close()

torch.save(model.state_dict(), 'multilabel_classifier.pt')

def extract_mfcc(segment, sample_rate=48000, n_mfcc=13):
    if hasattr(segment, 'numpy'):
        segment = segment.numpy()
    segment = np.array(segment, dtype=np.float32)
    mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_mean = np.squeeze(mfccs_mean)
    return mfccs_mean 