
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        channel_attention = self.fc1(avg_pool)
        channel_attention = F.relu(channel_attention)
        channel_attention = self.fc2(channel_attention)
        channel_attention = self.sigmoid(channel_attention)
        return x * channel_attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute spatial attention by applying average pooling and max pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate both avg and max pooling results
        attention = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution and sigmoid to generate spatial attention map
        spatial_attention = self.conv1(attention)
        spatial_attention = self.sigmoid(spatial_attention)
        return x * spatial_attention  # Apply spatial attention


class AttentionFusionNet(nn.Module):
    def __init__(self, in_channels=3):
        super(AttentionFusionNet, self).__init__()
        # 通道注意力
        self.channel_attention = ChannelAttention(in_channels)
        # 空间注意力
        self.spatial_attention = SpatialAttention(kernel_size=7)
        # 输出层，将融合后的特征映射为一个通道
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        # 先进行通道注意力
        x = self.channel_attention(x)
        # 然后进行空间注意力
        x = self.spatial_attention(x)
        # 最后通过 1x1 卷积生成最终输出
        x = self.conv_out(x)
        return x

# # 输入的形状是 [B, 3, H, W]
# input_tensor = torch.randn(2, 3, 240, 240)
#
# # 创建模型实例
# model = AttentionFusionNet(in_channels=3)
#
# # 获取模型输出，形状为 [B, 1, H, W]
# output = model(input_tensor)
#
# print("Output shape:", output.shape)
