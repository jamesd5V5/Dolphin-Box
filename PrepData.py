import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PrepData(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spec, label = self.data[idx]
        spec = np.expand_dims(spec, axis=0)
        spec = torch.tensor(spec, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return spec, label