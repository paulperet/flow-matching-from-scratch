import torch.nn as nn
import torch

class SimpleNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 2048),
            nn.ReLU(),
            nn.Linear(2048, 3*32*32)
        )

    def forward(self, inputs, time):
        return self.net(inputs).reshape(-1, 3, 32, 32)