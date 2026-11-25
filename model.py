import torch.nn as nn
import torch

class SimpleNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3*32*32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3*32*32)
        )

    def forward(self, input):
        return self.net(input)

if __name__ == "__main__":
    model = SimpleNNModel()
    input = torch.randn(32, 32*32)
    output = model(input)
    print(f"input: {input.shape}, output: {output.shape}")