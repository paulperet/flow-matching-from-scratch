import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

from tqdm import tqdm

from models import SimpleNNModel, SimpleCNNModel, SimpleUNetModel
from transforms import transform

EPOCHS = 5
BATCH_SIZE = 1024
INPUT_SIZE = 3*32*32

# Flowers102 Dataset
dataset = torchvision.datasets.Food101('./data', transform=transform, download=True)
print(f"Dataset size: {len(dataset)}")

# Data loaders
training_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Device selected: {device}")

# Model
model = SimpleUNetModel().to(device)
if device == "cuda":
    model.compile()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Loss
loss_fn = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler(device=device)

# Progress Bar
progress_bar = tqdm(total=EPOCHS)

# Training loop
for epoch in range(EPOCHS):

    # Running Loss
    running_loss = 0.0

    for i, images in enumerate(training_loader):
        z, _ = images
        z = z.to(device)

        batch_size = z.shape[0]

        # Zero gradients
        optimizer.zero_grad()

        # Sample time t from uniform distribution [0,1]
        t = torch.rand(batch_size).unsqueeze(-1).to(device)

        # Sample noise from normal distribution
        epsilon = torch.normal(mean=torch.zeros(batch_size, 3, 32, 32), std=torch.ones(batch_size, 3, 32, 32)).to(device)

        # Add noise to inputs
        x = t.unsqueeze(-1).unsqueeze(-1) * z + (1 - t.unsqueeze(-1).unsqueeze(-1)) * epsilon

        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            # Get direction from vector field at time t
            outputs = model(x, t)
            loss = loss_fn(outputs, (z - epsilon))

        
        scaler.scale(loss).backward()

        # Update the model
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Print loss
        # print(f"Batch Loss: {loss.item()}")

    # Epoch Loss
    progress_bar.set_description(f"Epoch {epoch} loss: {running_loss / len(training_loader)}")
    progress_bar.update(1)

    # Save Model
    torch.save(model, 'model.pth')