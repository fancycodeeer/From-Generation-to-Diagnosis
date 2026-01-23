import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalAttentionWithPooling(nn.Module):
    def __init__(self, in_channels=6077, reduction_ratio=16):
        super(DirectionalAttentionWithPooling, self).__init__()

        self.reduction_ratio = reduction_ratio

        # 计算沿C（通道）维度的注意力
        self.fc_c = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 计算沿H（高度）维度的注意力
        self.fc_h = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 计算沿W（宽度）维度的注意力
        self.fc_w = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Adaptive average pooling for each direction
        self.avg_pool_c = nn.AdaptiveAvgPool2d(1)  # Pool across H and W for channel-wise attention
        self.avg_pool_h = nn.AdaptiveAvgPool2d((1, None))  # Pool across W for height-wise attention
        self.avg_pool_w = nn.AdaptiveAvgPool2d((None, 1))  # Pool across H for width-wise attention

    def forward(self, x):
        # x: [B, C, H, W]

        # 计算沿C（通道）维度的注意力
        pooled_c = self.avg_pool_c(x)  # Shape: [B, C, 1, 1]
        attention_c = self.fc_c(pooled_c)  # Apply attention to the pooled output
        x_c_attention = x * attention_c  # 通道方向的注意力加权

        # 计算沿H（高度）维度的注意力
        pooled_h = self.avg_pool_h(x_c_attention)  # Shape: [B, C, 1, W]
        attention_h = self.fc_h(pooled_h)  # Apply attention to the pooled output
        x_h_attention = x_c_attention * attention_h  # 高度方向的注意力加权

        # 计算沿W（宽度）维度的注意力
        pooled_w = self.avg_pool_w(x_h_attention)  # Shape: [B, C, H, 1]
        attention_w = self.fc_w(pooled_w)  # Apply attention to the pooled output
        x_w_attention = x_h_attention * attention_w  # 宽度方向的注意力加权

        return x_w_attention  # 最终加权后的输出