import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, input_channels=3, pretrained=False, model='18'):
        super().__init__()
        # Load a pretrained ResNet backbone
        if model == '18':
            backbone = models.resnet18(pretrained=pretrained)
        if model == '50':
            backbone = models.resnet50(pretrained=pretrained)
        if model == '101':
            backbone = models.resnet101(pretrained=pretrained)



        # Modify the first convolutional layer to accept input_channels
        # The original conv1 has in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        self.backbone = backbone
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=backbone.conv1.out_channels,
                kernel_size=backbone.conv1.kernel_size,
                stride=backbone.conv1.stride,
                padding=backbone.conv1.padding,
                bias=backbone.conv1.bias is not None
            )

        # Remove the final fully connected layer
        # backbone.fc is nn.Linear(2048, num_classes)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)  # Single logit output
        )

    def forward(self, x):
        # x: [B, input_channels, H, W]
        features = self.backbone(x)  # [B, 2048]
        logits = self.classifier(features)  # [B]
        return torch.sigmoid(logits)

# Example usage:
# model = ResNet50(input_channels=3)
# x = torch.randn(8, 3, 256, 256)
# outputs = model(x)  # [8]
# print(outputs)


#!/usr/bin/env python3
# resnet18_backbone.py
# Author: YourName
# ---------------------------------------------------------
# Minimal, dependency-free ResNet-18 backbone in PyTorch.
# ---------------------------------------------------------
class BasicBlock(nn.Module):
    """3 × 3 → BN → ReLU → 3 × 3 → BN (+shortcut)"""
    expansion = 1      # 输出通道放大倍数（Bottleneck=4）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18Backbone(nn.Module):
    """Standard ResNet-18 backbone (no FC / avg-pool head)."""
    def __init__(self, in_channels=3, num_class=3):
        super().__init__()

        # Stem ---------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels, 64,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages -------------------------------------------------------------
        self.layer1 = self._make_layer(64,  64,  2, stride=1)   # C2
        self.layer2 = self._make_layer(64,  128, 2, stride=2)   # C3
        self.layer3 = self._make_layer(128, 256, 2, stride=2)   # C4
        self.layer4 = self._make_layer(256, 512, 2, stride=2)   # C5

        # 初始化权重
        self._init_weights()

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(512*8*8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_class),  # Single logit output
        )

    # ---------------- private helpers --------------------------------------
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """
        构造一个 stage，由 `blocks` 个 BasicBlock 组成。
        第一个 block 负责下采样（stride>1 时）＋通道对齐。
        """
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(in_channels, out_channels,
                             stride=stride, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming 正态初始化，BN γ=1, β=0"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    # ---------------- forward ----------------------------------------------
    def forward(self, x):
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))   # /2
        x = self.maxpool(x)                      # /4

        # Stages
        c2 = self.layer1(x)   # /4
        c3 = self.layer2(c2)  # /8
        c4 = self.layer3(c3)  # /16
        c5 = self.layer4(c4)  # /32
        # 返回最高层特征；如需多层，可改为 return (c2, c3, c4, c5)
        return self.classifier(torch.flatten(c5, 1, -1))


# ---------------- quick test -----------------------------------------------
if __name__ == "__main__":
    model = ResNet18Backbone(in_channels=3)
    dummy = torch.randn(1, 3, 224, 224)
    out   = model(dummy)
    print("Output shape:", out.shape)  # torch.Size([1, 512, 7, 7])
