#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from dataset import SegmentationDataset
from models import UNet
import torchvision.transforms as transforms
import utils
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ---------------------------
# 训练和验证过程
# ---------------------------

def train(args):
    set_seed(args.seed)  # 设置种子

    # 设置设备：GPU优先
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备：", device)

    # 设置transform
    transform_train = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.Grayscale()
    ])
    transform = transforms.Compose([transforms.Grayscale()])

    # 创建训练和验证数据集
    train_dataset = SegmentationDataset(args.train_image_dir, args.train_mask_dir, transform_train)
    val_dataset = SegmentationDataset(args.val_image_dir, args.val_mask_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 初始化模型并移动到设备上
    model = UNet(n_channels=2, n_classes=1, bilinear=True)
    model.to(device)

    # 定义损失函数和优化器（BCEWithLogitsLoss 内置sigmoid）
    BCELoss = nn.BCEWithLogitsLoss()
    DiceLoss = utils.DiceLoss()
    FocalLoss = utils.FocalLoss(alpha=0.25, gamma=5, reduction='mean')
    IoULoss = utils.IoULoss()
    # BoundaryLoss = utils.BoundaryLoss(batchsize=args.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    best_val_loss = float('inf')

    # 开始训练
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(torch.cat([images, images], dim=1))
            loss = 0.25 * BCELoss(outputs, masks) + 1.75 * FocalLoss(outputs, masks) + IoULoss(outputs, masks) + DiceLoss(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # 在验证集上评估模型
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = DiceLoss(outputs, masks)
                running_val_loss += loss.item() * images.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)

        print("Epoch [{}/{}], Train Loss: {:.4f}".format(epoch, args.epochs, train_loss))
        print("Epoch [{}/{}], Val Loss: {:.4f}".format(epoch, args.epochs, val_loss))

        # # 保存验证集loss最低的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("保存最优模型，当前验证loss：{:.4f}".format(best_val_loss))

        # 保存
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.checkpoint_dir, '{}.pth'.format(epoch))
        torch.save(model.state_dict(), checkpoint_path)

# ---------------------------
# 主函数，解析命令行参数
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练U-Net用于二分类语义分割")
    parser.add_argument("--train_image_dir", type=str, default='./data/train/T1_4', help="训练图像目录")
    parser.add_argument("--train_mask_dir", type=str, default='./data/train/seg_4', help="训练mask目录")
    parser.add_argument("--val_image_dir", type=str, default='./data/test/T1_4', help="验证图像目录")
    parser.add_argument("--val_mask_dir", type=str, default='./data/test/seg_4',  help="验证mask目录")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/", help="模型保存目录")
    parser.add_argument("--batch_size", type=int, default=2, help="每个batch的图像数量")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    train(args)
