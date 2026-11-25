import torch
from models import SimpleNNModel
from euler import euler

PATH = 'model.pth'

model = torch.load(PATH, weights_only=False)
euler(model=model, number_steps=20)