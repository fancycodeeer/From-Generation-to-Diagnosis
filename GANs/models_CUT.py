import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
	def __init__(self, in_features):
		super(ResidualBlock, self).__init__()

		conv_block = [nn.ReflectionPad2d(1),
					  nn.Conv2d(in_features, in_features, 3),
					  nn.InstanceNorm2d(in_features),
					  nn.ReLU(inplace=True),
					  nn.ReflectionPad2d(1),
					  nn.Conv2d(in_features, in_features, 3),
					  nn.InstanceNorm2d(in_features)]

		self.conv_block = nn.Sequential(*conv_block)

	def forward(self, x):
		return x + self.conv_block(x)


class Discriminator(nn.Module):
	def __init__(self, input_nc):
		super(Discriminator, self).__init__()

		# A bunch of convolutions one after another
		model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
				 nn.LeakyReLU(0.2, inplace=True)]

		model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
				  nn.InstanceNorm2d(128),
				  nn.LeakyReLU(0.2, inplace=True)]

		model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
				  nn.InstanceNorm2d(256),
				  nn.LeakyReLU(0.2, inplace=True)]

		model += [nn.Conv2d(256, 512, 4, padding=1),
				  nn.InstanceNorm2d(512),
				  nn.LeakyReLU(0.2, inplace=True)]

		# FCN classification layer
		model += [nn.Conv2d(512, 1, 4, padding=1)]

		self.model = nn.Sequential(*model)

	def forward(self, x):
		x = self.model(x)
		# Average pooling and flatten
		return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Generator(nn.Module):
	def __init__(self, input_nc, output_nc, n_residual_blocks=5):
		super(Generator, self).__init__()

		# Initial convolution block
		self.initial = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(input_nc, 64, 7),
			nn.InstanceNorm2d(64),
			nn.ReLU(inplace=True)
		)

		# Downsampling blocks
		self.down_blocks = nn.ModuleList()
		in_features = 64
		for i in range(4):
			out_features = in_features * 2
			self.down_blocks.append(
				nn.Sequential(
					nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
					nn.InstanceNorm2d(out_features),
					nn.ReLU(inplace=True)
				)
			)
			in_features = out_features

		# Residual blocks
		self.res_blocks = nn.ModuleList()
		for _ in range(n_residual_blocks):
			self.res_blocks.append(ResidualBlock(in_features))

		# Upsampling blocks
		self.up_blocks = nn.ModuleList()
		out_features = in_features // 2
		for i in range(4):
			self.up_blocks.append(
				nn.Sequential(
					nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
					nn.InstanceNorm2d(out_features),
					nn.ReLU(inplace=True)
				)
			)
			in_features = out_features
			out_features = in_features // 2

		# Output layer
		self.output_layer = nn.Sequential(
			nn.ReflectionPad2d(3),
			nn.Conv2d(64, output_nc, 7),
			nn.Tanh()
		)

	def forward(self, x, encode_only=False):
		# 存储所有特征图
		features = []

		# Initial convolution
		x = self.initial(x)

		# Downsampling
		for i, down_block in enumerate(self.down_blocks):
			x = down_block(x)
			features.append(x)  # 下采样特征

		# 如果只需要编码部分的特征，在这里返回
		if encode_only:
			features = features[1:]
			return features

		# Residual blocks
		for res_block in self.res_blocks:
			x = res_block(x)

		# Upsampling
		for up_block in self.up_blocks:
			x = up_block(x)

		# Output layer
		output = self.output_layer(x)

		return output


def PatchNCELoss(batch_size, feat_q, feat_k):
	batch_size = batch_size
	nce_T = 0.1  # temperature参数
	cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
	mask_dtype = torch.bool

	# forward
	batchSize = feat_q.shape[0]
	dim = feat_q.shape[1]
	feat_k = feat_k.detach()

	# 计算正样本对的相似度
	l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
	l_pos = l_pos.view(batchSize, 1)

	# 重塑特征以计算所有负样本对
	batch_dim_for_bmm = batch_size
	feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
	feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
	npatches = feat_q.size(1)

	# 计算batch内所有样本对的相似度
	l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))  # [B, P, P]

	# 将对角线元素（自身与自身的相似度）mask掉
	diagonal = torch.eye(npatches, device=feat_q.device, dtype=mask_dtype)[None, :, :]
	l_neg_curbatch.masked_fill_(diagonal, -10.0)

	# 重塑负样本相似度矩阵
	# 先把每行除了对角线元素之外的值提取出来
	l_neg = []
	for i in range(l_neg_curbatch.size(0)):
		neg_i = l_neg_curbatch[i].masked_select(~diagonal[0])
		l_neg.append(neg_i)
	l_neg = torch.stack(l_neg, dim=0)
	l_neg = l_neg.view(batch_dim_for_bmm * npatches, -1)

	# 拼接正负样本的相似度并应用温度系数
	out = torch.cat((l_pos, l_neg), dim=1) / nce_T

	# 计算NCE loss
	loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))

	return loss


class PatchSampleF(nn.Module):
	def __init__(self, nc=256, num_patches=512):
		super().__init__()
		self.nc = nc  # number of channels
		self.num_patches = num_patches

	def forward(self, feats, num_patches=256, patch_ids=None):
		return_ids = []
		return_feats = []

		for feat_idx, feat in enumerate(feats):
			B, C, H, W = feat.shape
			feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)  # B, HW, C

			if num_patches > 0:

				if patch_ids is not None:
					patch_id = patch_ids[feat_idx].flatten()
				else:
					patch_id = torch.randperm(feat_reshape.shape[1], device=feat.device)
					patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # 随机采样patch

				x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # 采样特征
				# print(x_sample.size())
			else:
				x_sample = feat_reshape
				patch_id = []

			return_ids.append(patch_id)
			x_sample = F.normalize(x_sample, p=2, dim=1)

			if num_patches == 0:
				x_sample = x_sample.permute(0, 2, 1).flatten(0, 1)

			return_feats.append(x_sample)

		return return_feats, return_ids

# class ModelOptions:
# 	def __init__(self):
# 		self.nce_layers = [1, 2, 3, 4]  # 使用哪些层的特征
# 		self.num_patches = 256  # 采样的patch数量
# 		self.batch_size = 1
# 		self.nce_T = 0.07  # temperature参数
# 		self.lambda_NCE = 1.0  # NCE loss的权重

def PatchNCE(Generator, batch_size, real, fake):
	nce_layers = [2, 3, 4]  # 使用哪些层的特征
	num_patches = 512  # 采样的patch数量
	batch_size = batch_size
	lambda_NCE = 1.0  # NCE loss的权重

	netF = PatchSampleF(nc=256, num_patches=512)
	# netE = Generator_CUT(1, 1)  # 你需要定义自己的编码器网络
	netE = Generator  # 你需要定义自己的编码器网络

	# forward
	feat_q = netE(fake, encode_only=True)
	feat_k = netE(real, encode_only=True)

	feat_k_pool, sample_ids = netF(feat_k, num_patches, None)
	feat_q_pool, _ = netF(feat_q, num_patches, sample_ids)

	total_nce_loss = 0.0
	for f_q, f_k, nce_layer in zip(feat_q_pool, feat_k_pool, nce_layers):
		loss = PatchNCELoss(batch_size, f_q, f_k) * lambda_NCE
		total_nce_loss += loss.mean()

	return total_nce_loss / len(nce_layers)

# # 初始化模型
# G = Generator(1, 1)
#
# # 假设的输入
# real = torch.randn(4, 1, 256, 256)  # 真实图像
# fake = torch.randn(4, 1, 256, 256)  # 生成图像
#
# # 计算损失
# loss = PatchNCE(G, 4, real, fake)
# print(loss)
