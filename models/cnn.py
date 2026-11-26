import torch.nn as nn
import torch

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
            nn.Linear(192*4*4+1, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 192*4*4),
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