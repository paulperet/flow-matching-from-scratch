import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

from tqdm import tqdm

from models.tests import SimpleNNModel, SimpleCNNModel, SimpleUNetModel
from utils.transforms import transform
from utils.custom_dataset import CustomImageDataset, CustomImageDatasetBuffer

def train(model, batch_size, epochs, output_file='model.pth', num_workers=0, dataset_path=None):

    print("Loading dataset")
    # Flowers102 Dataset
    if dataset_path == None:
        dataset = torchvision.datasets.Food101('./data', transform=transform, download=True)
    else:
        dataset = CustomImageDatasetBuffer(dataset_path, transform)

    train_set, val_set = torch.utils.data.dataset.random_split(dataset, [len(dataset)-1000, 1000])
    print(f"Dataset size: {len(dataset)}, Train: {len(train_set)}, Val: {len(val_set)}, Number of workers: {num_workers}")

    # Data loaders
    training_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device selected: {device}")

    # Model
    if model == "mlp":
        model = SimpleNNModel().to(device)
    elif model == "cnn":
        model = SimpleCNNModel().to(device)
    elif model == "unet":
        model = SimpleUNetModel().to(device)
    else:
        raise ValueError("Model not implemented")
    
    if device == "cuda":
        model.compile()
    print(f"Model: {type(model).__name__}, Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

    # Loss
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device=device)

    # Progress Bar
    progress_bar = tqdm(total=epochs)

    # Max val loss tracking for saving model
    max_val_loss = float("inf")

    # Training loop
    for epoch in range(epochs):

        # Running Loss
        running_loss = 0.0

        for i, images in enumerate(training_loader):
            z, _ = images
            z = z.to(device)

            batch_size_ = z.shape[0]

            # Zero gradients
            optimizer.zero_grad()

            # Sample time t from uniform distribution [0,1]
            t = torch.rand(batch_size_).unsqueeze(-1).to(device)

            # Sample noise from normal distribution
            epsilon = torch.normal(mean=torch.zeros(batch_size_, 3, 32, 32), std=torch.ones(batch_size_, 3, 32, 32)).to(device)

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

        # Validation loss
        running_val = 0.0
        with torch.no_grad():
            for i, images in enumerate(validation_loader):
                z, _ = images
                z = z.to(device)
    
                batch_size_ = z.shape[0]

                # Sample time t from uniform distribution [0,1]
                t = torch.rand(batch_size_).unsqueeze(-1).to(device)
    
                # Sample noise from normal distribution
                epsilon = torch.normal(mean=torch.zeros(batch_size_, 3, 32, 32), std=torch.ones(batch_size_, 3, 32, 32)).to(device)
    
                # Add noise to inputs
                x = t.unsqueeze(-1).unsqueeze(-1) * z + (1 - t.unsqueeze(-1).unsqueeze(-1)) * epsilon

                outputs = model(x, t)
                loss = loss_fn(outputs, (z - epsilon))
                running_val += loss.item()


        # Epoch Loss
        progress_bar.set_description(f"Epoch {epoch} Train loss: {running_loss / len(training_loader)}, Val loss: {running_val / len(validation_loader)}")
        progress_bar.update(1)

        # Save Model
        if running_val < max_val_loss:
            torch.save(model, output_file)
            max_val_loss = running_val
