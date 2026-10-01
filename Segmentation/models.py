import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        hidden_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False), nn.SiLU(), nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class CoordinateAttention(nn.Module):

    def __init__(self, channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w))
        x_w = x_w.permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        out = identity * a_h * a_w
        return out

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_attention=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.SiLU(), nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.SiLU())
        self.use_attention = use_attention
        if use_attention:
            self.coord_att = CoordinateAttention(out_channels)
            self.channel_att = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_attention:
            x = self.coord_att(x)
            x = self.channel_att(x)
        return x

class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_attention=True):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, use_attention=use_attention)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class UpBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True, use_attention=True):
        super(UpBlock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.reduce = nn.Identity()
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, use_attention=use_attention)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class UNetCoordinateChannelAttention(nn.Module):

    def __init__(self, in_channels=4, num_classes=1, base_channels=32, bilinear=True, use_attention=True):
        super(UNetCoordinateChannelAttention, self).__init__()
        self.inc = ConvBlock(in_channels, base_channels, use_attention=use_attention)
        self.down1 = DownBlock(base_channels, base_channels * 2, use_attention=use_attention)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, use_attention=use_attention)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, use_attention=use_attention)
        self.down4 = DownBlock(base_channels * 8, base_channels * 16, use_attention=use_attention)
        self.up1 = UpBlock(in_channels=base_channels * 16, skip_channels=base_channels * 8, out_channels=base_channels * 8, bilinear=bilinear, use_attention=use_attention)
        self.up2 = UpBlock(in_channels=base_channels * 8, skip_channels=base_channels * 4, out_channels=base_channels * 4, bilinear=bilinear, use_attention=use_attention)
        self.up3 = UpBlock(in_channels=base_channels * 4, skip_channels=base_channels * 2, out_channels=base_channels * 2, bilinear=bilinear, use_attention=use_attention)
        self.up4 = UpBlock(in_channels=base_channels * 2, skip_channels=base_channels, out_channels=base_channels, bilinear=bilinear, use_attention=use_attention)
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

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
        logits = self.out_conv(x)
        return logits
if __name__ == '__main__':
    model = UNetCoordinateChannelAttention(in_channels=2, num_classes=1, base_channels=32, bilinear=True, use_attention=True)
    x = torch.randn(2, 2, 240, 240)
    y = model(x)
    print('Input shape:', x.shape)
    print('Output shape:', y.shape)
