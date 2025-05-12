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

data = get_all_spectrograms()

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    stratify=[label for _, label in data],
    random_state=42
)

class_counts = {0: 0, 1: 0, 2: 0}
for _, label in test_data:
    class_counts[label] += 1
print(f"Test set: {class_counts[0]} Whistles, {class_counts[1]} Clicks, {class_counts[2]} BPs")

def train(model, dataloader, optimizer, criterion, epochs=7):
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

train_dataset = PrepData(train_data)
test_dataset = PrepData(test_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train(model, train_loader, optimizer, criterion, epochs=7)

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

def plot_confusion_matrix(y_true, y_pred, class_names=['Whistles', 'Clicks', 'BPs']):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    print(cm)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

def visualize_activations(model, dataset, index=0):
    model.eval()
    x, _ = dataset[index]
    x = x.unsqueeze(0)
    model(x)

    for layer_name, activation in model.activations.items():
        num_filters = activation.shape[1]
        fig, axes = plt.subplots(1, min(6, num_filters), figsize=(15, 5))
        fig.suptitle(f'Activations from {layer_name}', fontsize=16)

        for i in range(min(6, num_filters)):
            axes[i].imshow(activation[0, i], cmap='viridis', aspect='auto')
            axes[i].axis('off')
            axes[i].set_title(f'Filter {i}')

        plt.tight_layout()
        plt.savefig(f"{layer_name}_activations.png", dpi=300)
        plt.show()

y_true, y_pred = get_predictions(model, test_loader)
plot_confusion_matrix(y_true, y_pred)

visualize_activations(model, test_dataset, 0)

torch.save(model.state_dict(), 'dolp_classifier.pt')
print("A new dolph_classifier.pt model as been created.")
