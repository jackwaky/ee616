import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,4096),
        nn.ReLU(),
        nn.Linear(4096,2048),
        nn.ReLU(),
        nn.Linear(2048,1024),
        nn.ReLU(),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.Linear(512,256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.Linear(32,16),
        nn.ReLU(),
        nn.Linear(16,10))

    def forward(self,x):
        return self.layers(x)