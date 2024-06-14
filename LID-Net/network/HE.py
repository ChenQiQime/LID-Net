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

class HEdown(nn.Module):
    def __init__(self, n_1, n_2):
        super(HEdown, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_2, n_2, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv = nn.Conv2d(2*n_2, n_2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n_2, n_2, kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(n_1, n_2))

    def forward(self, x):
        x = self.maxpool_conv(x)
        y = x
        x1 = self.conv2(x)
        x2 = self.conv1(x)
        x3 = x1+x2
        x4 = torch.cat((y, x3),dim=1)
        x5 = self.relu(self.conv(x4))
        return x5


class HEup(nn.Module):
    def __init__(self, n_1, n_2):
        super(HEup, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.conv11 = nn.Conv2d(n_2, n_2, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(n_2, n_2, kernel_size=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv2 = nn.Conv2d(n_2, n_2, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(2*n_2, n_2, kernel_size=3, stride=1, padding=1)
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
        #print(x2.shape)
        x1 = self.conv3(x1)
        x3 = torch.cat([x2, x1], dim=1)
        x4 = self.conv(x3)

        y = x4
        x5 = self.conv2(x4)
        x6 = self.conv1(x4)
        x7 = x5 + x6

        x8 = torch.cat((y, x7), dim=1)
        x9 = self.relu(self.conv(x8))
        return x9






# input=torch.randn(results, 3, 320, 240)
# input1=torch.randn(results, 3, 640, 480)
#
# m=HEup(8,3)
# out=m(input,input1)
#
# print(out.shape)
