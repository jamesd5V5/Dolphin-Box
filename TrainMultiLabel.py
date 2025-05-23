import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from DataExtraction import get_all_spectrograms_with_pca
from PrepData import PrepData
from MultiLabelCNN import MultiLabelCNN
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare data
data = get_all_spectrograms_with_pca()
dataset = PrepData(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Convert single labels to multi-hot encoding
def convert_to_multi_hot(labels, num_classes=3):
    multi_hot = torch.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        multi_hot[i, label] = 1
    return multi_hot

# Convert labels to multi-hot encoding
labels = [label for _, label in data]
multi_hot_labels = convert_to_multi_hot(labels)

# Calculate class weights for balanced training
class_counts = multi_hot_labels.sum(dim=0)
total_samples = len(labels)
pos_weight = torch.tensor([2.0, 1.5, 2.0])  # [Whistle, Click, BP]
print("Class weights:", pos_weight)

# Initialize model and training components
model = MultiLabelCNN(input_size=50)
# Use CrossEntropyLoss for classification and BCEWithLogitsLoss for confidence
class_criterion = nn.CrossEntropyLoss(weight=pos_weight)
conf_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop
num_epochs = 40
best_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_confidences = []
    
    for inputs, labels in dataloader:
        # Convert labels to multi-hot encoding
        multi_hot = convert_to_multi_hot(labels)
        
        optimizer.zero_grad()
        class_output, conf_output = model(inputs)
        
        # Calculate losses
        class_loss = class_criterion(class_output, labels)
        conf_loss = conf_criterion(conf_output, multi_hot)
        
        # Combined loss (you can adjust the weights)
        loss = class_loss + 0.5 * conf_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        running_loss += loss.item()
        
        # Store predictions and labels for metrics
        with torch.no_grad():
            # Get classification predictions
            _, preds = torch.max(class_output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Get confidence scores
            confidences = torch.sigmoid(conf_output)
            all_confidences.extend(confidences.cpu().numpy())
    
    epoch_loss = running_loss/len(dataloader)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')
    
    # Calculate and print metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Calculate per-class metrics
    for i, class_name in enumerate(['Whistle', 'Click', 'BP']):
        class_preds = all_preds == i
        class_labels = all_labels == i
        accuracy = (class_preds == class_labels).mean()
        precision = (class_preds & class_labels).sum() / (class_preds.sum() + 1e-6)
        recall = (class_preds & class_labels).sum() / (class_labels.sum() + 1e-6)
        print(f'{class_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        print(f'{class_name} - Average Confidence: {all_confidences[:, i].mean():.4f}')
    
    # Update learning rate
    scheduler.step(epoch_loss)
    
    # Save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_multilabel_classifier.pt')

# Load best model for final evaluation
model.load_state_dict(torch.load('best_multilabel_classifier.pt'))
model.eval()

# Final evaluation
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

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_confidences = np.array(all_confidences)

# Print classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Whistle', 'Click', 'BP']))

# Plot confusion matrices for each class
class_names = ['Whistle', 'Click', 'BP']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (ax, class_name) in enumerate(zip(axes, class_names)):
    cm = multilabel_confusion_matrix(all_labels, all_preds)[i]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{class_name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
plt.tight_layout()
plt.savefig('multilabel_confusion_matrices.png')
plt.close()

# Save the final model
torch.save(model.state_dict(), 'multilabel_classifier.pt') 