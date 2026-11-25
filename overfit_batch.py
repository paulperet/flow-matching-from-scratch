import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

from models import SimpleNNModel, SimpleCNNModel
from transforms import transform

EPOCHS = 1000
BATCH_SIZE = 1

# Flowers102 Dataset
dataset = torchvision.datasets.Flowers102('./data', transform=transform, download=True)
print(f"Dataset size: {len(dataset)}")

# Model
model = SimpleNNModel()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Loss
loss_fn = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(EPOCHS):

    # Running Loss
    running_loss = 0.0

    z, _ = dataset[0]
    z = z.unsqueeze(0)

    batch_size = z.shape[0]

    # Zero gradients
    optimizer.zero_grad()

    # Sample time t from uniform distribution [0,1]
    t = torch.rand(batch_size).unsqueeze(-1)

    # Sample noise from normal distribution
    epsilon = torch.normal(mean=torch.zeros(batch_size, 3, 8, 8), std=torch.ones(batch_size, 3, 8, 8))

    # Add noise to inputs
    x = t.unsqueeze(-1).unsqueeze(-1) * z + (1 - t.unsqueeze(-1).unsqueeze(-1)) * epsilon

    # Get vector field at time t
    outputs = model(x, t)

    loss = loss_fn(outputs, (z - epsilon))
    loss.backward()

    # Update the model
    optimizer.step()

    running_loss += loss.item()

    # Epoch Loss
    print(f"Epoch {epoch} loss: {running_loss}")

# Save Model
torch.save(model, 'model.pth')