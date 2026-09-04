import torch
import torch.nn as nn
import torch.nn.functional as F

def gn(ch):
    for g in [32, 16, 8, 4, 2, 1]:
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)

class SEBlock(nn.Module):

    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, mid, 1, bias=True), nn.SiLU(inplace=True), nn.Conv2d(mid, ch, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)

class CoordAttention(nn.Module):

    def __init__(self, ch, reduction=32):
        super().__init__()
        mid = max(8, ch // reduction)
        self.conv1 = nn.Conv2d(ch, mid, kernel_size=1, bias=False)
        self.norm = gn(mid)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, ch, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mid, ch, kernel_size=1, bias=True)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.norm(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return identity * a_h * a_w

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, use_se=True, use_coord=True):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), gn(out_ch), nn.SiLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), gn(out_ch), nn.SiLU(inplace=True))
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.coord = CoordAttention(out_ch) if use_coord else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        x = self.coord(x)
        return x

class Down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = (skip.size(2) - x.size(2), skip.size(3) - x.size(3))
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNetSECoord(nn.Module):

    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.inc = ConvBlock(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16, base * 8, base * 8)
        self.up2 = Up(base * 8, base * 4, base * 4)
        self.up3 = Up(base * 4, base * 2, base * 2)
        self.up4 = Up(base * 2, base, base)
        self.outc = nn.Conv2d(base, out_ch, 1)
        nn.init.constant_(self.outc.bias, -6.0)

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
        return self.outc(x)
