import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# ---------------------------------- ResNet18 Backbone ----------------------------------
class BasicBlock_res(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(BasicBlock_res(self.in_planes, planes, stride))
        self.in_planes = planes * BasicBlock_res.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock_res(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x



# ---------------------------
# 定义U-Net模型结构
# ---------------------------
class CoordinateAttention(nn.Module):
	def __init__(self, in_channels, reduction=16):
		"""
		Coordinate Attention Module
		:param in_channels: 输入特征图的通道数
		:param reduction: 通道数压缩比例，用于降低维度
		"""
		super(CoordinateAttention, self).__init__()

		self.in_channels = in_channels
		self.reduction = reduction

		# 横向（水平）和纵向（垂直）坐标注意力
		self.horizontal_pool = nn.AdaptiveAvgPool2d((1, None))  # 水平方向池化
		self.vertical_pool = nn.AdaptiveAvgPool2d((None, 1))  # 垂直方向池化

		self.fc = nn.Sequential(
			nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
			nn.SiLU(),
			nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
		)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		"""
		:param x: 输入特征图，形状为 (batch_size, channels, height, width)
		"""
		batch_size, C, H, W = x.size()

		# 横向池化，计算水平注意力图
		horizontal_att = self.horizontal_pool(x)
		horizontal_att = self.fc(horizontal_att)
		horizontal_att = horizontal_att.view(batch_size, C, 1, W)

		# 纵向池化，计算垂直注意力图
		vertical_att = self.vertical_pool(x)
		vertical_att = self.fc(vertical_att)
		vertical_att = vertical_att.view(batch_size, C, H, 1)

		# 生成最终的坐标注意力图
		att_map = horizontal_att + vertical_att
		att_map = self.sigmoid(att_map)

		# 加权输入特征图
		out = x * att_map
		return out

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
        self.conv_1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1, stride=1)
        self.conv_2 = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv_2(F.relu(self.conv_1(x)))

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

        self.outconv_hm = OutConv(64, n_classes)
        # self.outconv_wh = OutConv(64, 2)

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

        logits_hm = self.outconv_hm(x)
        # logits_wh = self.outconv_wh(x)
        return logits_hm



# ----------------------------- Model -----------------------------
class CenterNet(nn.Module):
    def __init__(self, n_channels=1, num_classes=1):
        super().__init__()
        self.backbone = UNet(n_channels, num_classes)

    def forward(self, x):
        logits_hm = self.backbone(x)
        hm = torch.sigmoid(logits_hm)
        # off = self.off_head(feat)
        return hm