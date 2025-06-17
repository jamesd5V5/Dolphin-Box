import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelCNN(nn.Module):
    def __init__(self, input_size=63):
        super(MultiLabelCNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)

        self.class_head = nn.Linear(32, 4)
        self.confidence_head = nn.Linear(32, 4)
        
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
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        self.activations['fc3'] = x.detach().cpu()
        
        class_output = self.class_head(x)
        confidence_output = torch.sigmoid(self.confidence_head(x))
        
        return class_output, confidence_output 