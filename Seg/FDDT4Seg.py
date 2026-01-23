#!/usr/bin/env python
# -*- coding: utf-8 -*-
# torchrun --nproc_per_node=8 FDDT4Seg.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import argparse
from dataset import SegmentationDataset
from models import UNet, Generator, Discriminator
import torchvision.transforms as transforms
import utils
import random
import numpy as np
import itertools
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from utils.utils import weights_init


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_distributed():
    # 设置 GPU 设备
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # 初始化分布式训练
    dist.init_process_group(backend='nccl', init_method='env://')
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def train(args):
    set_seed(args.seed)  # 设置种子
    device = setup_distributed()  # 设置分布式设备

    rank = dist.get_rank()
    is_main = rank == 0

    if is_main:
        print("using device:", rank)

    # 数据增强
    transform_train = transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.Grayscale()
    ])
    transform = transforms.Compose([transforms.Grayscale()])

    # 数据集 & 分布式采样器
    train_dataset = SegmentationDataset(args.train_image_dir, args.train_mask_dir, args.train_image_t1ce_dir, transform_train)
    val_dataset = SegmentationDataset(args.val_image_dir, args.val_mask_dir, transform=transform)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    # 模型 & DDP封装
    segment = UNet(n_channels=1, n_classes=1, bilinear=True).cuda()
    segment = DDP(segment, device_ids=[device])
    translator = Generator(1, 1).cuda()
    translator.load_state_dict(torch.load('./models/cyc_fddt4seg//fddt_netG_A2B.pth'))
    translator = DDP(translator, device_ids=[device])

    # 冻结G
    for p in translator.parameters():
        p.requires_grad = False

    # 损失函数
    BCELoss = nn.BCEWithLogitsLoss()
    DiceLoss = utils.DiceLoss()
    FocalLoss = utils.FocalLoss(alpha=0.25, gamma=5, reduction='mean')
    IoULoss = utils.IoULoss()


    # 设置优化器
    optimizer_seg = optim.Adam(segment.parameters(), lr=args.lr, betas=(0.5, 0.999))
    lr_scheduler_seg = torch.optim.lr_scheduler.LambdaLR(optimizer_seg,  lr_lambda=utils.LambdaLR(args.epochs, 0, 50).step)
    best_val_loss = float('inf')


    for epoch in range(args.epochs):
        segment.train()

        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        running_loss_fake = 0.0

        for index, (images, masks, _) in enumerate(train_loader):
            images = images.cuda()
            masks = masks.cuda()

            # 分割real T1
            # optimizer_seg.zero_grad()
            # outputs = segment(images)
            # loss = 0.25 * BCELoss(outputs, masks) + 1.75 * FocalLoss(outputs, masks) + IoULoss(outputs, masks) + DiceLoss(outputs, masks)
            # loss.backward()
            # running_loss = running_loss + loss.item() * images.size(0)
            # optimizer_seg.step()


            # 分割fake T1ce
            optimizer_seg.zero_grad()
            with torch.no_grad():
                fake_T1ce = translator((images - 0.5) * 2.0)
            outputs_fake_T1ce = segment((fake_T1ce + 1.0)*0.5)
            loss_fake_T1ce = 1.75 * FocalLoss(outputs_fake_T1ce, masks) + 0.25 * BCELoss(outputs_fake_T1ce, masks) + IoULoss(outputs_fake_T1ce, masks) + DiceLoss(outputs_fake_T1ce, masks)
            loss_fake_T1ce.backward()
            running_loss_fake = running_loss_fake + loss_fake_T1ce.item() * images.size(0)
            optimizer_seg.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_loss_fake = running_loss_fake / len(train_loader.dataset)

        # 验证（只主进程打印/保存）
        segment.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda()
                masks = masks.cuda()
                fake_T1ce = translator((images - 0.5) * 2.0)
                outputs = segment((fake_T1ce + 1.0)*0.5)
                loss = DiceLoss(outputs, masks)
                running_val_loss = running_val_loss + loss.item() * images.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)

        if is_main:
            print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | fake Loss: {train_loss_fake:.4f}")
            # 保存模型
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(segment.module.state_dict(), os.path.join(args.checkpoint_dir, "best_model_{}.pth".format(epoch)))
                print(f"Save Best Epoch!!!, Val Loss: {best_val_loss:.4f}")

        # 更新学习率
        lr_scheduler_seg.step()

    print(f"Save Best Epoch!!!, Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    # T1/Flair 分割用的是seg_all
    parser = argparse.ArgumentParser(description="分布式训练U-Net用于语义分割")
    parser.add_argument("--train_image_dir", type=str, default='./data/train/T2')
    parser.add_argument("--train_image_t1ce_dir", type=str, default='./data/train/Flair')
    parser.add_argument("--train_mask_dir", type=str, default='./data/train/seg_all')
    parser.add_argument("--val_image_dir", type=str, default='./data/test/T2')
    parser.add_argument("--val_mask_dir", type=str, default='./data/test/seg_all')
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/fddt4seg/cyc_t2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
    args = parser.parse_args()

    train(args)
