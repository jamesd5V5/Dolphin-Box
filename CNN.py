import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)

        # Use dummy forward pass to find the flattened size
        self._to_linear = None
        self._get_flattened_size()

        self.fc1 = nn.Linear(36800, 64)
        self.fc2 = nn.Linear(64, 2)

    def _get_flattened_size(self):
        # Run dummy input through conv layers to get final shape
        x = torch.zeros(1, 1, 200, 94)  # [batch, channel, mel, time]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        #print("Shape before flatenning:", x.shape) #([16, 32, 23, 50])
        x = x.view(x.size(0), -1)
        #print("Shape After flatenning:", x.shape) #([16, 36800])
        x = F.relu(self.fc1(x))
        return self.fc2(x)

