import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BasicBlock(nn.Module):
	"""
	这是 ResNet-18 中的基本残差块，包含两个 3x3 卷积层。
	"""

	def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
		super(BasicBlock, self).__init__()

		# 第一个 3x3 卷积层
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)

		# 第二个 3x3 卷积层
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		# 用于匹配输入和输出的通道数（在 stride 不为 1 或输入和输出通道数不同的情况下）
		self.shortcut = nn.Sequential()
		if stride != 1 or in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		# Dropout层
		self.dropout = nn.Dropout(p=0.3)



	def forward(self, x: Tensor) -> Tensor:
		# 残差连接 + ReLU 激活
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.dropout(out)  # 在卷积层后应用 Dropout
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)  # 添加残差
		out = F.relu(out)  # 激活
		return out

class Bottleneck(nn.Module):
    """
    ResNet-50 的 Bottleneck Block
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(Bottleneck, self).__init__()

        # 1x1 卷积层，减少维度
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 卷积层，恢复维度
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        # 用于匹配输入和输出的通道数
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

        # Dropout层
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)  # 激活
        return out

class ResNet18(nn.Module):
	"""
	ResNet-18 模型，包括 4 个阶段，每个阶段包含多个残差块。
	"""

	def __init__(self, num_classes: int = 1000, dim=1):
		super(ResNet18, self).__init__()

		# 初始卷积层
		self.conv1 = nn.Conv2d(dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		# 残差块阶段
		self.layer1 = self._make_layer(64, 64, 2, stride=1)  # 2个基本残差块
		self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 2个基本残差块
		self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 2个基本残差块
		self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 2个基本残差块

		# 全局平均池化层
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		# 全连接层
		self.fc = nn.Linear(512, num_classes)

		# Dropout层
		self.dropout = nn.Dropout(p=0.3)

	def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
		"""
		生成一个残差块层（包含多个基本残差块）。
		"""
		layers = []
		layers.append(BasicBlock(in_channels, out_channels, stride))
		for _ in range(1, num_blocks):
			layers.append(BasicBlock(out_channels, out_channels, stride=1))
		return nn.Sequential(*layers)

	def forward(self, x: Tensor) -> Tensor:
		# 初始卷积层
		x = self.relu(self.bn1(self.conv1(x)))
		x = self.maxpool(x)

		# 通过各个残差块层
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		# 全局平均池化
		x = self.avgpool(x)
		x = torch.flatten(x, 1)  # 展平为 [B, 512]

		# 全连接层进行分类
		x = self.fc(x)
		return x

class ResNet50(nn.Module):
    """
    ResNet-50 模型，包括 4 个阶段，每个阶段包含多个 Bottleneck Block。
    """
    def __init__(self, num_classes: int = 1000, dim: int = 1):
        super(ResNet50, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv2d(dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差块阶段
        self.layer1 = self._make_layer(64, 64, 3, stride=1)  # 3个Bottleneck块
        self.layer2 = self._make_layer(256, 128, 4, stride=2)  # 4个Bottleneck块
        self.layer3 = self._make_layer(512, 256, 6, stride=2)  # 6个Bottleneck块
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)  # 3个Bottleneck块

        # 全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        生成一个残差块层（包含多个 Bottleneck Block）。
        """
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels * 4, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # 初始卷积层
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # 通过各个残差块层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平为 [B, 2048]

        # 通过全连接层
        x = self.fc(x)
        return x

# 测试模型
if __name__ == "__main__":
	model = ResNet50(num_classes=2).cuda()  # 创建一个ResNet-18模型，假设有1000个类别
	input_tensor = torch.randn(8, 1, 240, 240).cuda() # 假设输入是8个图像，大小为224x224
	output = model(input_tensor)
	print(f"Output shape: {output.shape}")
