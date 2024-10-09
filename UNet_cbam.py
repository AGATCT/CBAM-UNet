import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):   # the code from https://zhuanlan.zhihu.com/p/102035273 sets ratio = 16
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class cbam(nn.Module):
    def __init__(self, planes):
        super(cbam,self).__init__()
        self.ca = ChannelAttention(planes)# planes是feature map的通道个数
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x) * x  # 广播机制
        x = self.sa(x) * x  # 广播机制
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.cbam = cbam(3)
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = DoubleConv(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # torch.Size([16, 3, 256, 256])
        cbam_x = self.cbam(x)
        down1 = self.down1(cbam_x)
        # torch.Size([16, 32, 128, 128])
        pool1 = self.pool1(down1)
        # torch.Size([16, 32, 128, 128])
        down2 = self.down2(pool1)
        # torch.Size([16, 64, 128, 128])
        pool2 = self.pool2(down2)
        # torch.Size([16, 64, 64, 64])
        down3 = self.down3(pool2)
        # torch.Size([16, 128, 64, 64])
        pool3 = self.pool3(down3)
        # torch.Size([16, 128, 32, 32])
        down4 = self.down4(pool3)
        # torch.Size([16, 256, 32, 32])
        pool4 = self.pool4(down4)
        # torch.Size([16, 256, 16, 16])

        middle = self.middle(pool4)
        # torch.Size([16, 512, 16, 16])

        up1 = self.up1(middle)
        # torch.Size([16, 256, 32, 32])
        concat1 = torch.cat([down4, up1], dim=1)
        # torch.Size([16, 512, 32, 32])
        upconv1 = self.upconv1(concat1)
        # torch.Size([16, 256, 32, 32])

        up2 = self.up2(upconv1)
        concat2 = torch.cat([down3, up2], dim=1)
        upconv2 = self.upconv2(concat2)

        up3 = self.up3(upconv2)
        concat3 = torch.cat([down2, up3], dim=1)
        upconv3 = self.upconv3(concat3)

        up4 = self.up4(upconv3)
        concat4 = torch.cat([down1, up4], dim=1)
        upconv4 = self.upconv4(concat4)

        out = self.out_conv(upconv4)
        return out