import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvLayer(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class UpSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvLayer(in_channels, out_channels)

    def forward(self, x1, x2):
        # Upsample x1 and adjust size to match x2
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # Concatenate and convolve
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.downconv1 = DownSampleLayer(in_channels, 64)  # Adjusted input channels here
        self.downconv2 = DownSampleLayer(64, 128)
        self.downconv3 = DownSampleLayer(128, 256)
        self.downconv4 = DownSampleLayer(256, 512)

        self.bottleneck = DoubleConvLayer(512, 1024)

        self.upconv1 = UpSampleLayer(1024, 512)
        self.upconv2 = UpSampleLayer(512, 256)
        self.upconv3 = UpSampleLayer(256, 128)
        self.upconv4 = UpSampleLayer(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
       # print(f"Input shape: {x.shape}")  # Debug input shape
        down1, p1 = self.downconv1(x)
       # print(f"After downconv1: {down1.shape}")
        down2, p2 = self.downconv2(p1)
       # print(f"After downconv2: {down2.shape}")
        down3, p3 = self.downconv3(p2)
       # print(f"After downconv3: {down3.shape}")
        down4, p4 = self.downconv4(p3)
       # print(f"After downconv4: {down4.shape}")

        bottle = self.bottleneck(p4)

       # print(f"After bottle: {bottle.shape}")
        up1 = self.upconv1(bottle, down4)
       # print(f"After upconv1: {up1.shape}")
        up2 = self.upconv2(up1, down3)
       # print(f"After upconv2: {up2.shape}")
        up3 = self.upconv3(up2, down2)
       # print(f"After upconv3: {up3.shape}")
        up4 = self.upconv4(up3, down1)
       # print(f"After upconv4: {up4.shape}")

        out = self.out(up4)
        return out


if __name__ == "__main__":
    model = UNet(in_channels=1, num_classes=1)
    dummy_input = torch.randn(1, 1, 256, 256)  # Example input
    output = model(dummy_input)
    print("Output shape:", output.shape)
