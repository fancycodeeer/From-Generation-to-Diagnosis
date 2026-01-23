import torch
import torch.nn as nn
import torch.nn.functional as F


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
