import torch.nn as nn
import torch

from models.cnn import SimpleCNNModel
from models.mlp import SimpleNNModel
from models.unet import SimpleUNetModel

if __name__ == "__main__":
    models = [SimpleNNModel(), SimpleCNNModel(), SimpleUNetModel()]
    for model in models:
        input = torch.randn(1, 3, 32, 32)
        time = torch.randn(1, 1)
        output = model(input, time)
        print(f"{type(model).__name__}, size: {(sum(p.numel() for p in model.parameters())/1e6):.3f}M, input: {input.shape}, output: {output.shape}")