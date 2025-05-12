import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(36800, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        """
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d((2, 2)),
            nn.Dropout(0.3)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d((2, 2)),
            nn.Dropout(0.4)
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 
        """

        self.fc2 = nn.Linear(64, 3)

        self.activations = {}

    def _get_flattened_size(self):
        x = torch.zeros(1, 1, 200, 94)  # [batch, channel, mel, time]
        x = F.relu(self.bn1(self.conv1(x))) 
        x = F.relu(self.bn2(self.conv2(x)))
        self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        self.activations['conv1'] = x.detach().cpu()

        x = F.relu(self.bn2(self.conv2(x)))
        self.activations['conv2'] = x.detach().cpu()

        '''
        x = self.conv_block3(x)
        self.activations['conv_block3'] = x.detach().cpu()

        x = self.gap(x)
        '''
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        return self.fc2(x)

