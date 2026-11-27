import torch.nn as nn
import torch

class SimpleCNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, padding=1, stride=2), # 3x32x32 -> 12x16x16
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=48, kernel_size=3, padding=1, stride=2), # 12*16*16 -> 48*8*8
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=192, kernel_size=3, padding=1, stride=2), # 48*8*8 -> 192*4*4
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(192*4*4, 2048),
            nn.ReLU(),
            nn.Linear(2048, 192*4*4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=torch.Size([192, 4, 4])),

            nn.ConvTranspose2d(in_channels=192, out_channels=48, kernel_size=3, padding=1, stride=2, output_padding=1), # 192*4*4 -> 48*8*8 
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=48, out_channels=12, kernel_size=3, padding=1, stride=2, output_padding=1), # 48*8*8 -> 12*16*16
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=3, padding=1, stride=2, output_padding=1), # 12*16*16 -> 3x32x32
        )

    def forward(self, inputs, time):
        return self.net(inputs)