import torch.nn as nn
import torch

class SimpleNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*32*32, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3*32*32)
        )

    def forward(self, inputs, time):
        return self.net(inputs).reshape(-1, 3, 32, 32)

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

class SimpleUNetModel(nn.Module):
    def __init__(self, batch_size=1):
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

        self.buffer1 = torch.empty((batch_size, 512, 8, 8))
        self.buffer2 = torch.empty((batch_size, 256, 16, 16))
        self.buffer3 = torch.empty((batch_size, 128, 32, 32))

    def forward(self, inputs, time):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        bottleneck = self.bottleneck(conv3)

        self.buffer1[:, conv3.size()[1]:, :, :] = conv3
        self.buffer1[:, :conv3.size()[1], :, :] = bottleneck

        deconv1 = self.deconv1(self.buffer1)

        self.buffer2[:, conv2.size()[1]:, :, :] = conv2
        self.buffer2[:, :conv2.size()[1], :, :] = deconv1

        deconv2 = self.deconv2(self.buffer2)

        self.buffer3[:, conv1.size()[1]:, :, :] = conv1
        self.buffer3[:, :conv1.size()[1], :, :] = deconv2

        deconv3 = self.deconv3(self.buffer3)

        return deconv3

if __name__ == "__main__":
    models = [SimpleNNModel(), SimpleCNNModel(), SimpleUNetModel(batch_size=512)]
    for model in models:
        input = torch.randn(512, 3, 32, 32)
        time = torch.randn(512, 1)
        output = model(input, time)
        print(f"{type(model).__name__} input: {input.shape}, output: {output.shape}")