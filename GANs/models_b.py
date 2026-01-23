import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torch
import math

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
        self.attention = ChannelAttention(in_features)

    def forward(self, x):
        return x + self.attention(self.conv_block(x))


class ConvUpsample(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, scale_factor=2):
        super(ConvUpsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


# class Generator(nn.Module):
#     def __init__(self, input_nc, output_nc, n_residual_blocks=9):
#         super(Generator, self).__init__()
#
#         # Initial convolution block
#         model = [   nn.ReflectionPad2d(3),
#                     nn.Conv2d(input_nc, 64, 7),
#                     nn.InstanceNorm2d(64),
#                     nn.ReLU(inplace=True) ]
#
#         # Downsampling
#         in_features = 64
#         out_features = in_features*2
#         for _ in range(2):
#             model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
#                         nn.InstanceNorm2d(out_features),
#                         nn.ReLU(inplace=True) ]
#             in_features = out_features
#             out_features = in_features*2
#
#         # Residual blocks
#         for _ in range(n_residual_blocks):
#             model += [ResidualBlock(in_features)]
#
#         # Upsampling
#         out_features = in_features//2
#         for _ in range(2):
#             model += [  # nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
#                         ConvUpsample(in_features, out_features),
#                         nn.InstanceNorm2d(out_features),
#                         nn.ReLU(inplace=True) ]
#             in_features = out_features
#             out_features = in_features//2
#
#         # Output layer
#         model += [  nn.ReflectionPad2d(3),
#                     nn.Conv2d(64, output_nc, 7),
#                     nn.Tanh() ]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    ChannelAttention(128)]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    ChannelAttention(256)]

        model += [  nn.Conv2d(256, 512, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),
                    ChannelAttention(512)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x).squeeze(-1)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            ChannelAttention(128)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            ChannelAttention(256)
        )

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks += [ResidualBlock(256)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            ChannelAttention(128)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),  # 256 because of skip connection
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            ChannelAttention(64)
        )


        # Output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, output_nc, 7),  # 128 because of skip connection
            nn.Tanh()
        )

    def forward(self, x):
        # Initial convolution
        x1 = self.initial(x)

        # Downsampling
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Residual blocks
        x3 = self.res_blocks(x3)

        # Upsampling + skip connections
        x = self.up1(x3)
        x = self.up2(torch.cat([x, x2], 1))  # Skip connection
        x = self.output(torch.cat([x, x1], 1))  # Skip connection

        return x



class Adaptation(nn.Module):
    def __init__(self, input_nc=2):
        super(Adaptation, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 32, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x, y):
        output1 = self.model(x)
        output2 = self.model(y)
        output = torch.cat((output1, output2), dim=1)
        output = F.softmax(output, dim=1)

        return output[:, 0, :, :], output[:, 1, :, :]


class NonLinearLayer(nn.Module):
    def __init__(self, input_nc):
        super(NonLinearLayer, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, input_nc*3, 3),
            nn.InstanceNorm2d(input_nc * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc*3, input_nc*16, 3),
            nn.InstanceNorm2d(input_nc*64),
            nn.LeakyReLU(0.2, inplace=True),
            #
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(input_nc * 2, input_nc * 2, 3),
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(input_nc * 2, input_nc * 2, 3),
            # nn.InstanceNorm2d(input_nc * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            #
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(input_nc * 2, input_nc * 2, 3),
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(input_nc * 2, input_nc * 2, 3),
            # nn.InstanceNorm2d(input_nc * 2),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.ReflectionPad2d(1),
            # nn.Conv2d(input_nc * 2, input_nc, 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc*16, input_nc*3, 3),
            nn.InstanceNorm2d(input_nc),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc*3, input_nc, 3),
            nn.InstanceNorm2d(input_nc),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# x = torch.randn(32, 1, 240, 121)
# y = torch.randn(32, 1, 240, 121)
# model = Adaptation(input_nc=1)
# z1, z2 = model(x, y)
# print(z1.shape)