import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DataExtraction import get_all_spectrograms_with_pca
from PrepData import PrepData
from CNN import CNN
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = get_all_spectrograms_with_pca()
dataset = PrepData(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

labels = [label for _, label in data] #class weights
class_counts = np.bincount(labels)
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()  # Normalize
print("Class weights:", weights)

model = CNN(input_size=50) #50 PCA Componnents
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%')

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Whistle', 'Click', 'BP'], yticklabels=['Whistle', 'Click', 'BP'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

torch.save(model.state_dict(), 'dolp_classifier.pt') 