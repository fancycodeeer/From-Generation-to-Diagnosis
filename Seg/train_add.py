#!/usr/bin/env python
# -*- coding: utf-8 -*-
# torchrun --nproc_per_node=4 train_FDDT.py

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
    # segment.load_state_dict(torch.load('./checkpoints/T1/best_model.pth'))
    segment = DDP(segment, device_ids=[device])
    translator = Generator(1, 1).cuda()
    translator.load_state_dict(torch.load(args.trans_dir))
    translator = DDP(translator, device_ids=[device])
    # discriminator = Discriminator(1).cuda()
    # discriminator = DDP(discriminator, device_ids=[device])


    # 冻结G
    for p in translator.parameters():
        p.requires_grad = False

    # 损失函数
    BCELoss = nn.BCEWithLogitsLoss()
    DiceLoss = utils.DiceLoss()
    FocalLoss = utils.FocalLoss(alpha=0.25, gamma=5, reduction='mean')
    IoULoss = utils.IoULoss()
    L2loss = torch.nn.MSELoss()

    # 设置优化器
    optimizer_seg = optim.Adam(segment.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # optimizer_g = optim.Adam(translator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    # optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    # lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=utils.LambdaLR(args.epochs, 0, 30).step)
    # lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=utils.LambdaLR(args.epochs, 0, 20).step)
    lr_scheduler_seg = torch.optim.lr_scheduler.LambdaLR(optimizer_seg, lr_lambda=utils.LambdaLR(args.epochs, 0, 30).step)
    best_val_loss = float('inf')

    # 定义GAN所需变量
    fake_T1_buffer = utils.ReplayBuffer()
    target_real = Variable(torch.Tensor(args.batch_size).fill_(1.0), requires_grad=False).cuda()
    target_fake = Variable(torch.Tensor(args.batch_size).fill_(0.0), requires_grad=False).cuda()

    for epoch in range(args.epochs):
        segment.train()
        # translator.train()
        # discriminator.train()

        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        running_loss_fake = 0.0
        running_loss_g = 0.0
        running_loss_d = 0.0


        for index, (images, masks, images_t1ce) in enumerate(train_loader):
            images = images.cuda()
            masks = masks.cuda()
            images_t1ce = images_t1ce.cuda()
            images_t1ce = transform(images_t1ce)

            # if is_main:
            #     fake_T1_xx = translator(images_t1ce)
            #     fake_T1_xx = 0.5 * (fake_T1_xx.data + 1.0)
            #     save_image(fake_T1_xx, './output/trans/' + str(index) + '.png')

            # 分割real T1
            # optimizer_seg.zero_grad()
            # outputs = segment(images)
            # loss = 0.25 * BCELoss(outputs, masks) + 1.75 * FocalLoss(outputs, masks) + IoULoss(outputs, masks) + DiceLoss(outputs, masks)
            # loss.backward()
            # running_loss = running_loss + loss.item() * images.size(0)
            # optimizer_seg.step()


            # 分割fake T1
            optimizer_seg.zero_grad()
            # optimizer_g.zero_grad()
            fake_T1ce = 0.0#translator((images-0.5)*2.0)
            outputs_fake_T1 = segment(images+(fake_T1ce + 1.0)*0.5)
            loss_fake_T1 = 1.75 * FocalLoss(outputs_fake_T1, masks) + 0.25 * BCELoss(outputs_fake_T1, masks)
            loss_fake_T1.backward()
            running_loss_fake = running_loss_fake + loss_fake_T1.item() * images.size(0)
            optimizer_seg.step()
            # optimizer_g.step()

            # 微调GAN
            # optimizer_g.zero_grad()
            # fake_T1_gan = translator(images_t1ce)
            # # pred_fake_T1 = discriminator(fake_T1_gan)
            # # g_loss = L2loss(pred_fake_T1, target_real)
            # instance_loss = L2loss(fake_T1_gan, (images - 0.5)*2.0)
            # loss_g = instance_loss
            # loss_g.backward()
            # running_loss_g = running_loss_g + loss_g.item() * images.size(0)
            # optimizer_g.step()

            # optimizer_d.zero_grad()
            # # Real loss
            # pred_real_T1 = discriminator(images)
            # loss_D_real = L2loss(pred_real_T1, target_real)
            # # Fake loss
            # fake_A = fake_T1_buffer.push_and_pop(fake_T1_gan)
            # pred_fake_T1 = discriminator(fake_A.detach())
            # loss_D_fake = L2loss(pred_fake_T1, target_fake)
            #
            # # Total loss
            # loss_d = (loss_D_real + loss_D_fake) * 0.5
            # loss_d.backward()
            # running_loss_d = running_loss_d + loss_d.item() * images.size(0)
            # optimizer_d.step()


        train_loss = running_loss / len(train_loader.dataset)
        train_loss_fake = running_loss_fake / len(train_loader.dataset)
        train_loss_g = running_loss_g / len(train_loader.dataset)
        train_loss_d = running_loss_d / len(train_loader.dataset)

        # 验证（只主进程打印/保存）
        segment.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda()
                masks = masks.cuda()
                fake_T1ce = 0.0#translator((images - 0.5) * 2.0)
                outputs = segment(images+(fake_T1ce + 1.0)*0.5)
                loss = DiceLoss(outputs, masks)
                running_val_loss = running_val_loss + loss.item() * images.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)

        if is_main:
            print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | fake Loss: {train_loss_fake:.4f} | G-Loss: {train_loss_g:.4f} | D-Loss: {train_loss_d:.4f}")
            # 保存模型
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(segment.module.state_dict(), os.path.join(args.checkpoint_dir, "best_model_{}.pth".format(epoch)))
                print(f"Save Best Epoch!!!, Val Loss: {best_val_loss:.4f}")

            # torch.save(segment.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}.pth"))
            # torch.save(translator.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}" + '_G.pth'))
            # torch.save(discriminator.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}" + '_D.pth'))

        # 更新学习率
        # lr_scheduler_G.step()
        # lr_scheduler_D.step()
        lr_scheduler_seg.step()

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分布式训练U-Net用于语义分割")
    parser.add_argument("--train_image_dir", type=str, default='./data/train/T2')
    parser.add_argument("--train_image_t1ce_dir", type=str, default='./data/train/Flair')
    parser.add_argument("--train_mask_dir", type=str, default='./data/train/seg_all')
    parser.add_argument("--val_image_dir", type=str, default='./data/test/T2')
    parser.add_argument("--val_mask_dir", type=str, default='./data/test/seg_all')
    parser.add_argument("--trans_dir", type=str, default='./models/cyc_fddt4seg/fdit_netG_t2.pth')
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/fddt4seg/base/")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
    args = parser.parse_args()

    train(args)
