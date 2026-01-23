import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class FSeSimLoss(nn.Module):
	def __init__(self):
		super(FSeSimLoss, self).__init__()

		# 加载预训练的VGG16
		self.vgg = models.vgg16(pretrained=True).features.eval()

		# 目标特征层的索引
		self.target_layers = {
			'relu1_2': '4',
			'relu2_2': '9',
			'relu3_3': '16',
			'relu4_3': '23'
		}

		# 冻结VGG参数
		for param in self.vgg.parameters():
			param.requires_grad = False

		self.criterion = nn.L1Loss()

	def compute_gram(self, x):
		b, ch, h, w = x.size()
		f = x.view(b, ch, w * h)
		f_T = f.transpose(1, 2)
		G = f.bmm(f_T) / (b * h * w * ch)
		return G

	def get_features(self, x):
		features = {}
		current_feat = x

		# 直接前向传播并获取中间特征
		for name, layer in self.vgg.named_children():
			current_feat = layer(current_feat)
			if name in self.target_layers.values():
				layer_name = [k for k, v in self.target_layers.items() if v == name][0]
				features[layer_name] = current_feat

		return features

	def forward(self, x, y):
		# repeat channel
		b, c, h, w = x.size()
		if c != 3 and c == 1:
			x = x.repeat(1, 3, 1, 1)
			y = y.repeat(1, 3, 1, 1)

		# 提取特征
		x_features = self.get_features(x)
		y_features = self.get_features(y)

		# 计算风格损失
		style_loss = 0.0
		for layer in self.target_layers.keys():
			x_gram = self.compute_gram(x_features[layer])
			y_gram = self.compute_gram(y_features[layer])
			style_loss += self.criterion(x_gram, y_gram)

		return style_loss