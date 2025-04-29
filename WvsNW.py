import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from CNN import CNN
from PrepData import PrepData
from DataExtraction import get_all_spectrograms

# Load data
data = get_all_spectrograms()

# Train-test split
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    stratify=[label for _, label in data],
    random_state=42
)

# Print class distribution in the test set
class_counts = {0: 0, 1: 0, 2: 0}
for _, label in test_data:
    class_counts[label] += 1
print(f"Test set: {class_counts[0]} Clicks, {class_counts[1]} Whistles, {class_counts[2]} BPs")

# Define training function
def train(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, Accuracy={accuracy:.2f}%")

# Create datasets and loaders
train_dataset = PrepData(train_data)
test_dataset = PrepData(test_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize model (make sure your CNN outputs 3 classes)
model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train the model
train(model, train_loader, optimizer, criterion, epochs=10)

# Function to get predictions
def get_predictions(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)

# Updated plot function for 3 classes
def plot_confusion_matrix(y_true, y_pred, class_names=['Click', 'Whistle', 'Burst Pulse']):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

# Evaluate model
y_true, y_pred = get_predictions(model, test_loader)
plot_confusion_matrix(y_true, y_pred)
