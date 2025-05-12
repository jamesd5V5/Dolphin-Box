import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        
        # Depthwise separable conv block
        self.depthwise = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
        self.pointwise = nn.Conv2d(8, 16, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classification layers
        self.fc = nn.Linear(16, 3)
        
        self.activations = {}

    def forward(self, x):
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        self.activations['conv1'] = x.detach().cpu()
        
        # Depthwise separable conv
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = F.relu(self.bn2(x))
        self.activations['conv2'] = x.detach().cpu()
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        return self.fc(x)

