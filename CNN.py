import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size=10):  # default 10, but 50 for now
        super(CNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(32, 3)
        
        self.activations = {}

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        self.activations['fc1'] = x.detach().cpu()
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        self.activations['fc2'] = x.detach().cpu()
        
        return self.fc3(x)

