import string
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from RNN import RNNNet
from torch.utils.data import DataLoader

from Dataset import SpectorgramDataset
from DataExtraction import get_all_spectrograms

#Load Dataset
data = get_all_spectrograms()
dataset = SpectorgramDataset(data)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

input_size = dataset[0][0].shape[1] # number of mel bens
model = RNNNet(input_size=input_size, hidden_size=128, num_classes=2)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for specs, labels in loader:
        # specs: (batch, seq_len, input_size)
        outputs = model(specs)  # (batch, num_classes)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

'''
def get_random_seq():

    #seq_len = 128

    #t = np.arange(0, seq_len)
    #a = 2*np.pi*1.0/seq_len
    #b = 2*np.pi*np.random.rand()*5
    #seq = np.sin(a*t+b)
    return seq

def get_input_and_target():
    seq = get_random_seq()
    input = torch.tensor(seq[:-1]).float().view(-1,1,1)
    target = torch.tensor(seq[1:]).float().view(-1,1,1)
    return input, target

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_size = 1
        self.hidden_size = 100
        self.output_size = 1

        self.rnn = nn.RNNCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        hidden = self.rnn(input, hidden)
        output = self.linear(hidden)

        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size) #Intial hidden Sate, 1 means batch size = 1
    
net = Net()

def train_step(net, opt, input, target):
    seq_len = input.shape[0] #Get sequence length of current input
    hidden = net.init_hidden() #intial hidden state
    net.zero_grad() #clear the gradient
    loss = 0 #intial loss

    for t in range(seq_len):
        output, hidden = net(input[t], hidden)
        loss += loss_func(output, target[t])

    loss.backward()
    opt.step() #update weights

    return loss / seq_len #return avg loss w.r.t sequence length

def eval_step(net, predicted_len=100):
    hidden = net.init_hidden()
    init_seq = get_random_seq()
    init_input = torch.tensor(init_seq).float().view(-1,1,1)
    predicted_seq = []

    for t in range(len(init_seq) - 1): #use intial pts on curve t build hidden state
        output, hidden = net(init_input[t], hidden)

    input = init_input[-1] #set current input as last chracter of the intial string

    for t in range(predicted_len):
        output, hideen = net(input, hidden)
        predicted_seq.append(output.item())
        input = output

    return init_seq, predicted_seq

#Training
iters = 200
print_iters = 10
all_losses = []
loss_sum = 0

opt = torch.optim.Adam(net.parameters(), lr=0.005)
loss_func = nn.MSELoss()

for i in range(iters):
    input, target = get_input_and_target()
    loss = train_step(net, opt, input, target) #Calc loss
    loss_sum += loss #accumulate loss

    if i% print_iters == print_iters - 1:
        print('iter:{}/{} loss:{}'.format(i, iters, loss_sum / print_iters))
        all_losses.append(loss_sum.cpu().detach().numpy() / print_iters)
        loss_sum = 0

plt.figure(figsize=(16,10))
plt.xlabel("iters")
plt.ylabel("loss")
plt.plot(all_losses)
plt.savefig("Training Loss Curve")
'''
