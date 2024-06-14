from network.unet_parts import *
from network.HR import HRdown
from network.HR import HRup
from network.HE import HEdown
from network.HE import HEup


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 4))
        self.down1 = (HEdown(4, 8))
        self.down2 = (HRdown(8, 16, 160, 120))
        self.down3 = (HEdown(16, 32))
        self.down4 = (HRdown(32, 64, 40, 30))
        self.up1 = HEup(64, 32)
        self.up2 = (HRup(32, 16, 160, 120))
        self.up3 = (HEup(16, 8))
        self.up4 = (HRup(8, 4, 640, 480))
        self.outc = (OutConv(4, n_classes))

    def forward(self, x):
        y = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x, y)
        return logits




# input=torch.randn(results, 3, 640, 480)
#
# m=UNet()
# out=m(input)
#
# print(out.shape)

