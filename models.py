import torch.nn as nn
import torch

class SimpleNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3*8*8+1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3*8*8)
        )

    def forward(self, images, time):
        images = torch.flatten(images, start_dim=1)
        inputs = torch.cat((images, time), dim=-1)
        return self.net(inputs).reshape((-1, 3, 8, 8))

class SimpleCNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1, stride=2), # 3x32x32 -> 12x16x16
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=48, kernel_size=3, padding=1, stride=2), # 12*16*16 -> 48*8*8
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=192, kernel_size=3, padding=1, stride=2), # 48*8*8 -> 192*4*4
            nn.ReLU(),
            nn.Flatten(),
        )
        self.nn = nn.Sequential(
            nn.Linear(192*4*4+1, 192*4*4),
            nn.ReLU(),
        )

        self.conv_transpose = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=torch.Size([192, 4, 4])),
            nn.ConvTranspose2d(in_channels=192, out_channels=48, kernel_size=3, padding=1, stride=2, output_padding=1), # 192*4*4 -> 48*8*8 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=48, out_channels=12, kernel_size=3, padding=1, stride=2, output_padding=1), # 48*8*8 -> 12*16*16
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=3, padding=1, stride=2, output_padding=1), # 12*16*16 -> 3x32x32
        )

    def forward(self, inputs, time):
        flattened = self.conv(inputs)
        flatten_and_time = torch.cat((flattened, time), dim=-1)

        outputs = self.nn(flatten_and_time)
        outputs = self.conv_transpose(outputs)

        return outputs

if __name__ == "__main__":
    models = [SimpleNNModel(), SimpleCNNModel()]
    for model in models:
        input = torch.randn(1, 3, 32, 32)
        time = torch.randn(1, 1)
        output = model(input, time)
        print(f"{type(model).__name__} input: {input.shape}, output: {output.shape}")