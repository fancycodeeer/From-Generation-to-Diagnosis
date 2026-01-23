"""
纯 Python 版本的 fused_act.py，不依赖 C++/CUDA 编译。
原始作用：将 bias 加入输入，再使用 LeakyReLU 激活，并乘以一个缩放因子。
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=math.sqrt(2)):
    """
    计算：F.leaky_relu(input + bias) * scale

    Args:
        input (Tensor): 输入张量，形状通常为 (N, C, H, W)。
        bias (Tensor or None): 一维偏置，形状为 (C,)，如果不为 None，则加到 input 上。
        negative_slope (float): LeakyReLU 的负斜率。
        scale (float): 缩放因子。

    Returns:
        Tensor: 经过加 bias、LeakyReLU 激活和缩放后的输出。
    """
    if bias is not None:
        # 将 bias 扩展到 (1, C, 1, 1) 后与 input 相加
        bias = bias.view(1, -1, 1, 1)
        input = input + bias
    return F.leaky_relu(input, negative_slope=negative_slope) * scale

class FusedLeakyReLU(nn.Module):
    """
    FusedLeakyReLU 模块：包含一个可学习的偏置参数。
    使用 fused_leaky_relu 函数实现前向计算。
    """
    def __init__(self, channel, negative_slope=0.2, scale=math.sqrt(2)):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
