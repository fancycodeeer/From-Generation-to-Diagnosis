import torch
import torch.nn as nn


def postprocess_output(x):
	x = x * 0.5
	x = x + 0.5
	x = x + 255
	return x


class CrossModalAttentionNetwork(nn.Module):
	def __init__(self, threshold_minmax=[0.5, 4.5, 5, 25], num_heads=2):
		super(CrossModalAttentionNetwork, self).__init__()

		self.threshold_minmax = threshold_minmax

		# 空间特征提取模块
		self.spatial_extractor = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=1),
			nn.InstanceNorm2d(16),
			nn.AvgPool2d(2),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.InstanceNorm2d(32),
			nn.ReLU(),
			nn.AvgPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.InstanceNorm2d(64),
			nn.ReLU(),
			nn.AvgPool2d(2),
		)

		# 频域特征提取模块
		self.frequency_extractor = nn.Sequential(
			nn.Conv2d(2, 16, kernel_size=3, padding=1),
			nn.InstanceNorm2d(16),
			nn.AvgPool2d(2),
			nn.ReLU(),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.InstanceNorm2d(32),
			nn.ReLU(),
			nn.AvgPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.InstanceNorm2d(64),
			nn.ReLU(),
			nn.AvgPool2d(2),
		)

		# 多头自注意力模块：跨模态注意力
		self.cross_modal_attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads)

		# 层次化特征融合模块
		self.feature_fusion = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=3, padding=1),
			nn.InstanceNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 8, kernel_size=3, padding=1),
			nn.InstanceNorm2d(8),
			nn.ReLU(),
		)

		# 阈值预测网络
		self.fc_h = nn.Sequential(
			nn.Flatten(),
			nn.Linear(32 * 32 * 8, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()  # 输出[0, 1]之间的值
		)

		self.fc_l = nn.Sequential(
			nn.Flatten(),
			nn.Linear(32 * 32 * 8, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()  # 输出[0, 1]之间的值
		)

	def get_frequency_features(self, x):
		# if torch.isnan(x).any() or torch.isinf(x).any():
		# 	raise ValueError("Input contains NaN or Infinity!")

		# 计算傅里叶变换，获取频域的幅度和相位信息
		fft = torch.fft.fft2(postprocess_output(x))
		fft_shift = torch.fft.fftshift(fft)
		magnitude = torch.log(torch.abs(fft_shift) + 1)
		phase = torch.angle(fft_shift)
		freq_features = torch.cat([magnitude, phase], dim=1)
		return freq_features

	def forward(self, x):
		# if torch.isnan(x).any() or torch.isinf(x).any():
		# 	raise ValueError("Input contains NaN or Infinity!")

		# 1. 提取空间特征
		spatial_features = self.spatial_extractor(x)

		# 2. 提取频域特征
		freq_features = self.get_frequency_features(x)
		frequency_features = self.frequency_extractor(freq_features)

		# 3. 跨模态特征融合：通过多头注意力机制进行空间和频域特征的交互
		spatial_features_flat = spatial_features.view(spatial_features.size(0), -1, spatial_features.size(1))
		frequency_features_flat = frequency_features.view(frequency_features.size(0), -1, frequency_features.size(1))

		# 将空间和频域特征堆叠在一起，作为多头注意力的输入
		combined_features = torch.cat([spatial_features_flat, frequency_features_flat], dim=1)
		attention_output, _ = self.cross_modal_attention(combined_features, combined_features, combined_features)
		dynamic_features = attention_output.view(spatial_features.size(0), spatial_features.size(1) + frequency_features.size(1), spatial_features.size(2), spatial_features.size(3))

		# 4. 层次化特征融合
		fused_features = self.feature_fusion(dynamic_features)

		# 5. 预测阈值D
		D_h_normalized = self.fc_h(fused_features)
		D_h = self.threshold_minmax[0] + (self.threshold_minmax[1] - self.threshold_minmax[0]) * D_h_normalized

		D_l_normalized = self.fc_l(fused_features)
		D_l = self.threshold_minmax[2] + (self.threshold_minmax[3] - self.threshold_minmax[2]) * D_l_normalized

		return D_h, D_l


# x = torch.randn(1, 1, 256, 256).cuda()
# model = CrossModalAttentionNetwork().cuda()
# h, l = model(x)
#
# print(h)
# print(l)