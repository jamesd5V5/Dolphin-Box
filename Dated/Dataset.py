import torch
from torch.utils.data import Dataset

class SpectorgramDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        spec, label = self.data[index]
        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return spec_tensor, label_tensor