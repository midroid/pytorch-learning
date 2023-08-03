import torch
import torch.nn as nn
import torch.nn.functional as F

# Option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # nn.Sigmoid
        # nn.Softmax
        # nn.Tanh
        # nn.LeakyReLU

    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out
    

# Option 2 (use activation functions directly in forward pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)



    def forward(self, x):
        # torch.sigmoid
        # torch.tanh
        # F.relu 
        # F.leaky_relu
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
    
input_size = 28 * 28
hidden_size = 5
model1 = NeuralNet(input_size=input_size, hidden_size=hidden_size)
model2 = NeuralNet2(input_size=input_size, hidden_size=hidden_size)
criterion = nn.BCELoss()