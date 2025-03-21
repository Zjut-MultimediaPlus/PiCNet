import torch
from torch import nn
import torch.nn.functional as F
import loralib as lora

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, mid_channel=None):
        super(DoubleConv, self).__init__()
        # self.in_channel = in_channel
        if not mid_channel:
            mid_channel = out_channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        # print('in_channel:', self.in_channel)
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channel, out_channel, mid_channel=int(in_channel // 2))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv_l(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, mid_channel=None):
        super(DoubleConv_l, self).__init__()
        # self.in_channel = in_channel
        if not mid_channel:
            mid_channel = out_channel
        self.conv = nn.Sequential(
            lora.Conv2d(in_channel, mid_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(True),
            lora.Conv2d(mid_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

    def forward(self, x):
        # print('in_channel:', self.in_channel)
        return self.conv(x)


class Down_l(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down_l, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv_l(in_channel, out_channel),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_l(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up_l, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv_l(in_channel, out_channel, mid_channel=int(in_channel // 2))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv_l(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv_l, self).__init__()
        self.conv = lora.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class Net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Net, self).__init__()
        self.inc = DoubleConv(in_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channel)

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)

        return out

class Net_2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Net_2, self).__init__()
        self.inc = DoubleConv(in_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channel)
        self.gamma = nn.Parameter(torch.zeros(1, out_channel, 1, 1), requires_grad=True)

        self.vup1 = Up(1024, 256)
        self.vup2 = Up(512, 128)
        self.vup3 = Up(256, 64)
        self.vup4 = Up(128, 64)
        self.voutc = OutConv(64, out_channel * 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        outs = self.outc(x) * self.gamma

        v = self.up1(x5, x4)
        v = self.up2(v, x3)
        v = self.up3(v, x2)
        v = self.up4(v, x1)
        outv = self.voutc(v)

        return outs, outv

class Net_l(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Net_l, self).__init__()
        self.inc = DoubleConv_l(in_channel, 64)
        self.down1 = Down_l(64, 128)
        self.down2 = Down_l(128, 256)
        self.down3 = Down_l(256, 512)
        self.down4 = Down_l(512, 512)

        self.up1 = Up_l(1024, 256)
        self.up2 = Up_l(512, 128)
        self.up3 = Up_l(256, 64)
        self.up4 = Up_l(128, 64)
        self.outc = OutConv_l(64, out_channel)
        self.gamma = nn.Parameter(torch.zeros(1, out_channel, 1, 1), requires_grad=True)

        self.vup1 = Up_l(1024, 256)
        self.vup2 = Up_l(512, 128)
        self.vup3 = Up_l(256, 64)
        self.vup4 = Up_l(128, 64)
        self.voutc = OutConv_l(64, out_channel * 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        outs = self.outc(x) * self.gamma

        v = self.up1(x5, x4)
        v = self.up2(v, x3)
        v = self.up3(v, x2)
        v = self.up4(v, x1)
        outv = self.voutc(v)

        return outs, outv

class Net_frz(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Net_frz, self).__init__()
        self.inc = DoubleConv(in_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channel)
        self.gamma = nn.Parameter(torch.zeros(1, out_channel, 1, 1), requires_grad=True)

        self.vup1 = Up(1024, 256)
        self.vup2 = Up(512, 128)
        self.vup3 = Up(256, 64)
        self.vup4 = Up(128, 64)
        self.voutc = OutConv(64, out_channel * 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        outs = self.outc(x) * self.gamma

        v = self.up1(x5, x4)
        v = self.up2(v, x3)
        v = self.up3(v, x2)
        v = self.up4(v, x1)
        outv = self.voutc(v)

        return outs, outv
# x = torch.randn((4, 9, 288, 288))
# net = Net_frz(9, 9)
# for name, params in net.named_parameters():
#     print(name)
#     params.requires_grad = False
#     print('name: {0},\t grad: {1}'.format(name, params.requires_grad))

# y = net(x)
# print(y.shape)
