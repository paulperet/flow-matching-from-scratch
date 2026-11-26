import torch
from utils.transforms import transform
import torchvision

import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 4

# Flowers102 Dataset
dataset = torchvision.datasets.Food101('./data', transform=transform, download=True)

# Data loaders
training_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# From Pytorch Docs
# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(training_loader)
images, labels = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=False)