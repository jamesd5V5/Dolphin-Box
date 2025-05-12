import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size=10):  # 10 PCA components
        super(CNN, self).__init__()
        
        # Fully connected layers for PCA input
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        
        # Final classification layer
        self.fc3 = nn.Linear(32, 3)
        
        self.activations = {}

    def forward(self, x):
        # Ensure input is 2D [batch_size, features]
        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        # First FC block
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        self.activations['fc1'] = x.detach().cpu()
        
        # Second FC block
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        self.activations['fc2'] = x.detach().cpu()
        
        # Classification
        return self.fc3(x)

