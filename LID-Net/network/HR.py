

import torch
from torch import nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * test"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.double_conv(x)

class HRdown(nn.Module):
    def __init__(self, n_1, n_2, h, w):
        super(HRdown, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_2, n_2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv = nn.Conv2d(2*n_2, n_2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n_2, n_2, kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(n_1, n_2))
        self.max_pool = nn.AdaptiveMaxPool2d((h, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, w))
        self.sig = nn.Sigmoid()
        self.conv11 = nn.Conv2d(n_2, n_2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.maxpool_conv(x)
        y = x
        x1 = self.sig(self.conv11(self.max_pool(y)))
        x2 = self.sig(self.conv11(self.avg_pool(y)))
        x3 = x1 * y
        x4 = y * x2
        x5 = x3 + x4
        x6 = self.relu(self.conv1(x5))
        return x6

class HRup(nn.Module):
    def __init__(self, n_1, n_2, h, w):
        super(HRup, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()

        self.conv11 = nn.Conv2d(n_2, n_2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(n_2, n_2, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.AdaptiveMaxPool2d((h, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, w))
        self.conv = DoubleConv(n_1, n_2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv3 = nn.Conv2d(n_1, n_2, kernel_size=1, stride=1, padding=0)

    def forward(self, x,x2):
        x1 = self.up(x)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # print(x1.shape)
        # print(x2.shape)
        x1 = self.conv3(x1)
        x3= torch.cat([x2, x1], dim=1)
        x3 = self.conv(x3)


        y = x3

        x4 = self.sig(self.conv11(self.max_pool(x3)))
        x5 = self.sig(self.conv11(self.avg_pool(x3)))
        x6 = x4*y
        x7 = y*x5
        x8 = x6+x7
        x9 = self.relu(self.conv1(x8))


        return x9

