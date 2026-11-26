import torch.nn as nn
import torch

class SimpleUNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1), # 3x32x32 -> 64x32x32
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), # 64x32x32 -> 64x32x32
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), # 64x32x32 -> 64x32x32
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # 64x32x32 -> 64x16x16
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1), # 64x16x16 -> 128x16x16
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1), # 128x16x16 -> 128x16x16
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1), # 128x16x16 -> 128x16x16
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # 128x16x16 -> 128x8x8
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1), # 128x8x8 -> 256x8x8
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1), # 256x8x8 -> 256x8x8
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1), # 256x8x8 -> 256x8x8
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # 256x8x8 -> 256x4x4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1), # 256x4x4 -> 512x4x4
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1), # 512x4x4 -> 512x4x4
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2, output_padding=1), # 512x4x4 -> 256x8x8
        )

        self.deconv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1), # 256x8x8 + 256x8x8 -> 256x8x8
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1), # 256x8x8 -> 256x8x8
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2, output_padding=1), #256x8x8 -> 128x16x16
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1), # 128x16x16 + 128x16x16 -> 128x16x16
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1), # 128x16x16 -> 128x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1), # 128x16x16 -> 64x32x32
        )

        self.deconv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1), # 64x32x32 + 64x32x32-> 64x32x32
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), # 64x32x32-> 64x32x32
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1), # 64x32x32-> 64x32x32
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, padding=0, stride=1), # 64x32x32-> 3x32x32
        )

    def forward(self, inputs, time):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        bottleneck = self.bottleneck(conv3)
        
        deconv1 = self.deconv1(torch.cat((conv3, bottleneck), dim=1))
        deconv2 = self.deconv2(torch.cat((conv2, deconv1), dim=1))
        deconv3 = self.deconv3(torch.cat((conv1, deconv2), dim=1))

        return deconv3