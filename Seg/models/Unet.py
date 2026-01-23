import torch
import torch.nn as nn
import torch.nn.functional as F
from .CoordinateAttention import CoordinateAttention
# ---------------------------
# 定义U-Net模型结构
# ---------------------------

# 双卷积模块：两层卷积 + 批归一化 + ReLU激活
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            CoordinateAttention(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

# 下采样模块：最大池化后接双卷积
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# 上采样模块：上采样/反卷积后与编码器对应层拼接，再接双卷积
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # 使用双线性插值上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # 使用反卷积上采样
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 调整 x1 的尺寸以与 x2 保持一致（必要时进行补零）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        # 拼接特征图（沿着通道维度）
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 输出层：1x1卷积将通道数映射为目标类别数
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# U-Net整体网络
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        n_channels: 输入图像的通道数（如RGB为3）
        n_classes: 输出类别数（对于二分类分割，可设为1）
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)     # 特征尺寸保持不变
        x2 = self.down1(x1)  # 下采样到1/2尺寸
        x3 = self.down2(x2)  # 1/4
        x4 = self.down3(x3)  # 1/8
        x5 = self.down4(x4)  # 1/16
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ---------------- 基础残差块 ----------------
class BasicBlock(nn.Module):
    """两个 3×3 Conv，保持通道数不变"""
    expansion = 1

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += identity                       # 残差相加
        out = F.relu(out, inplace=True)
        return out

# ---------------- 二分类 ResNet ----------------
class ResNetClassifier(nn.Module):
    """
    输入 : (B, 512, 15, 15)
    输出 : (B, 2)  —— 二分类 logits
    """
    def __init__(self, in_channels: int = 512,
                 num_blocks: int = 3,
                 hidden_dim: int = 256,
                 num_classes: int = 2):
        super().__init__()

        # 多个 BasicBlock 顺序堆叠
        blocks = [BasicBlock(in_channels) for _ in range(num_blocks)]
        self.backbone = nn.Sequential(*blocks)

        # 全局平均池化 + MLP 分类头
        self.pool = nn.AdaptiveAvgPool2d(1)        # → (B, C, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                          # (B, C)
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)                       # (B, 512, 15, 15)
        x = self.pool(x)                           # (B, 512, 1, 1)
        logits = self.classifier(x)                # (B, 2)
        return logits




if __name__ == '__main__':
    # model = UNet(1, 1)
    # x = torch.randn(1, 1, 256, 256)  # batch_size=1, 输入 256x256 图片
    # y = model(x)
    # print(y.shape)  # 输出应为 (1, 1, 256, 256)

    # 输入通道数为1，输出类别数为1
    n_channels = 2
    n_classes = 1

    # 构建UNet模型
    model = UNet(n_channels, n_classes)

    # 查看模型中的卷积层
    conv_layers = []


    def extract_conv_layers(module):
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d):  # 如果是卷积层
                conv_layers.append((child_name, child.in_channels, child.out_channels))
            # 递归遍历子模块
            if len(list(child.children())) > 0:
                extract_conv_layers(child)

    # 提取所有卷积层的信息
    extract_conv_layers(model)

    # 打印卷积层的输入通道和输出通道
    for layer in conv_layers:
        print(f"Layer: {layer[0]}, Input Channels: {layer[1]}, Output Channels: {layer[2]}")

    # 计算通道数总和
    total_channels = sum([layer[2] for layer in conv_layers])
    print(f"Total output channels from all Conv2d layers: {total_channels}")


