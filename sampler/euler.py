import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


INPUT_SIZE = 3*32*32

def euler(model, number_steps, samples=1, show_progression=False):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device selected: {device}")
    model.to(device)

    if show_progression:
        plt.figure(figsize = (samples, number_steps))
        gs = gridspec.GridSpec(samples, number_steps)
        gs.update(wspace=0.00, hspace=0.00) # set the spacing between axes. 
    
    t = 0
    step_size = 1 / number_steps


    # Generating n samples
    for sample in range(samples):
        # Gaussian noise / Initial state
        x = torch.normal(mean=torch.zeros(1, 3, 32, 32), std=torch.ones(1, 3, 32, 32)).to(device)

        # Denoising / Generating samples
        for i in range(number_steps):
            x += step_size * model(x, torch.tensor([t]).unsqueeze(0).to(device))

            if show_progression:
                ax = plt.subplot(gs[sample, i])
                ax.imshow(x.squeeze(0).detach().cpu().numpy().transpose(1,2,0).clip(0, 1))
                ax.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')

    # Remove batch dim
    x = x.squeeze(0)

    # Convert to numpy
    x = x.detach().cpu().numpy()
    x = x.transpose(1,2,0)

    if not show_progression:
        plt.imshow(x)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

if __name__ == "__main__":
    from models.tests import SimpleNNModel
    model = SimpleNNModel()
    euler(model=model, number_steps=10)