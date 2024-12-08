import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.pooling = pooling

        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.5)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = self.dropout(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(
            self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv3x3(self.out_channels, self.out_channels)

        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        from_down = self._crop_and_match(from_down, from_up)

        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), dim=1)
        else:
            x = from_up + from_down

        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.1)
        x = self.dropout(x)
        return x

    def _crop_and_match(self, down, up):
        _, _, h, w = up.size()
        return down[:, :, :h, :w]


class AudioDenoiserUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=5, start_filts=64, up_mode='transpose', merge_mode='concat'):
        super(AudioDenoiserUNet, self).__init__()

        if up_mode not in ('transpose', 'upsample'):
            raise ValueError(
                f"Invalid mode for upsampling: {up_mode}. Use 'transpose' or 'upsample'.")

        if merge_mode not in ('concat', 'add'):
            raise ValueError(
                f"Invalid mode for merging: {merge_mode}. Use 'concat' or 'add'.")

        if up_mode == 'upsample' and merge_mode == 'add':
            raise ValueError("Upsample with 'add' merge mode is incompatible.")

        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.up_mode = up_mode
        self.merge_mode = merge_mode

        self.down_convs = []
        self.up_convs = []

        # Encoder path
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # Decoder path
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=self.up_mode,
                             merge_mode=self.merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, out_channels=1)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def _crop_to_match(self, x, target):
        """
        Adjust tensor `x` to match the spatial dimensions of `target`.
        If `x` is smaller than `target`, pad it. If larger, crop it.
        """
        _, _, h_target, w_target = target.size()
        _, _, h, w = x.size()

        # Pad or crop height
        if h < h_target:
            pad_height = h_target - h
            # Pad height at the bottom
            x = F.pad(x, (0, 0, 0, 0, 0, pad_height))
        elif h > h_target:
            x = x[:, :, :h_target, :]

        # Pad or crop width
        if w < w_target:
            pad_width = w_target - w
            x = F.pad(x, (0, pad_width, 0, 0))  # Pad width on the right
        elif w > w_target:
            x = x[:, :, :, :w_target]

        return x

    def forward(self, x, target=None):
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)

        # Ensure output matches target size (if provided)
        if target is not None:
            x = self._crop_to_match(x, target)

        return x


if __name__ == "__main__":
    model = AudioDenoiserUNet(
        in_channels=1, out_channels=1, depth=5, merge_mode='concat')
    x = Variable(torch.FloatTensor(np.random.random((1, 1, 320, 320))))
    out = model(x)
    print(out.shape)
