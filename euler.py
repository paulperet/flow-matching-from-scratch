import torch
import matplotlib.pyplot as plt

INPUT_SIZE = 3*32*32

def euler(model, number_steps):
    t = 0
    step_size = 1 / number_steps

    # Gaussian noise / Initial state
    x = torch.normal(mean=torch.zeros(1, INPUT_SIZE), std=torch.ones(1, INPUT_SIZE))

    # Denoising / Generating samples
    for i in range(number_steps):
        x = x + step_size * model(x, torch.tensor([t]).unsqueeze(0))
        t = t+step_size

    # Final image
    x = torch.func.rearrange(x, "b (c h w) -> b c h w", c=3, h=32, w=32)

    # Remove batch dim
    x = x.squeeze(0)

    # Convert to numpy
    x = x.detach().numpy()
    x = x.transpose(1,2,0)

    plt.imshow(x)
    plt.show()

if __name__ == "__main__":
    from model import SimpleNNModel
    model = SimpleNNModel()
    euler(model=model, number_steps=10)