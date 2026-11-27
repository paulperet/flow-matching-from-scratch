import torch
from sampler.euler import euler

PATH = 'model.pth'

model = torch.load(PATH, weights_only=False, map_location="cpu")
euler(model=model, number_steps=10, samples=10, show_progression=True)