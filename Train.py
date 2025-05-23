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

# Add temperature parameter
TEMPERATURE = 1.0  # Default temperature, can be adjusted

data = get_all_spectrograms_with_pca()
dataset = PrepData(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

labels = [label for _, label in data] #class weights
class_counts = np.bincount(labels)
# Further adjust weights to give even more importance to whistles
weights = torch.tensor([2.0, 0.7, 0.8], dtype=torch.float)  # [Whistle, Click, BP]
weights = weights / weights.sum()  # Normalize
print("Class weights:", weights)

model = CNN(input_size=50) #50 PCA Componnents
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

num_epochs = 20  # Increased epochs
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # Apply temperature scaling to outputs before loss calculation
        scaled_outputs = outputs / TEMPERATURE
        loss = criterion(scaled_outputs, labels)
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        # Apply temperature scaling for predictions
        _, predicted = torch.max(scaled_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss/len(dataloader)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {100 * correct / total:.2f}%')
    
    # Update learning rate based on loss
    scheduler.step(epoch_loss)
    
    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_dolp_classifier.pt')

# Load best model for evaluation
model.load_state_dict(torch.load('best_dolp_classifier.pt'))
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        # Apply temperature scaling for evaluation
        scaled_outputs = outputs / TEMPERATURE
        _, predicted = torch.max(scaled_outputs.data, 1)
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