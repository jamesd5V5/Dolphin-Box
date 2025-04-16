import torch.nn as nn
import torch

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_classes=2):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        #x: (batch, seq_len, input_size)
        out, hidden = self.rnn(x)
        final_hidden = hidden.squeeze(0)
        out = self.fc(final_hidden)
        return out
