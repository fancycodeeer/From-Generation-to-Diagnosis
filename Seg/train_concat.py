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
from models import DirectionalAttentionWithPooling as Attention
import torchvision.transforms as transforms
import utils
import random
import numpy as np
import itertools
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from utils.utils import weights_init


os.environ["OMP_NUM_THREADS"] = "4"
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
    val_dataset = SegmentationDataset(args.val_image_dir, args.val_mask_dir, args.val_image_t1ce_dir, transform)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8, pin_memory=True)

    # 模型 & DDP封装
    segment = UNet(n_channels=2, n_classes=1, bilinear=True).cuda()
    segment = DDP(segment, device_ids=[device])
    segment_t = UNet(n_channels=2, n_classes=1, bilinear=True).cuda()
    # segment_t.load_state_dict(torch.load('./models/seg_teacher.pth'))
    segment_t = DDP(segment_t, device_ids=[device])
    translator = Generator(1, 1).cuda()
    translator.load_state_dict(torch.load('./models/299_netG_A2B.pth'))
    translator = DDP(translator, device_ids=[device])

    # discriminator_B = Discriminator(1).cuda()
    # discriminator_B.load_state_dict(torch.load('./models/99_D_B.pth'))
    # discriminator_B = DDP(discriminator_B, device_ids=[device])

    # attention = Attention(in_channels=2881, reduction_ratio=64).cuda()
    # attention = DDP(attention, device_ids=[device])

    # 冻结segment_t
    # for p in segment_t.parameters():
    #     p.requires_grad = False
    # for p in translator_t.parameters():
    #     p.requires_grad = False

    # 钩子函数：提取中间层特征
    features_s = []
    features_t = []
    features_detach_s = []
    features_detach_t = []
    def register_hooks(model, hook_fn):
        # 遍历模型的所有层并为卷积层注册钩子
        for name, module in model.named_modules():
            # if 'Down' in module.__class__.__name__ or 'Up' in module.__class__.__name__ or 'OutConv' in module.__class__.__name__:
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(hook_fn)

    def hook_fn_s(module, input, output):
        features_s.append(output)

    def hook_fn_t(module, input, output):
        features_t.append(output.detach())  # 将输出特征存储到列表中

    # def hook_fn_trans_s(module, input, output):
    #     features_trans_s.append(output)  # 将输出特征存储到列表中
    # def hook_fn_trans_t(module, input, output):
    #     features_trans_t.append(output)  # 将输出特征存储到列表中

    # 注册钩子
    register_hooks(segment, hook_fn=hook_fn_s)
    register_hooks(segment_t, hook_fn=hook_fn_t)
    # register_hooks(translator, hook_fn=hook_fn_trans_s)
    # register_hooks(translator_t, hook_fn=hook_fn_trans_t)

    # 损失函数
    BCELoss = nn.BCEWithLogitsLoss()
    DiceLoss = utils.DiceLoss()
    FocalLoss = utils.FocalLoss(alpha=0.25, gamma=5, reduction='mean')
    IoULoss = utils.IoULoss()
    L2loss = torch.nn.MSELoss()

    # 设置优化器
    optimizer_seg = optim.Adam(segment.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_seg_t = optim.Adam(segment_t.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(translator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    # optimizer_d = optim.Adam(discriminator_B.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=utils.LambdaLR(args.epochs, 0, 30).step)
    # lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=utils.LambdaLR(args.epochs, 0, 20).step)
    lr_scheduler_seg = torch.optim.lr_scheduler.LambdaLR(optimizer_seg, lr_lambda=utils.LambdaLR(args.epochs, 0, 30).step)
    lr_scheduler_seg_t = torch.optim.lr_scheduler.LambdaLR(optimizer_seg_t, lr_lambda=utils.LambdaLR(args.epochs, 0, 30).step)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        segment.train()
        translator.train()
        segment_t.train()
        # discriminator.train()

        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        running_loss_fake = 0.0
        running_loss_g = 0.0
        running_loss_tal = 0.0


        for index, (images, masks, images_t1ce) in enumerate(train_loader):
            images = images.cuda()
            masks = masks.cuda()
            images_t1ce = images_t1ce.cuda()
            images_t1ce = transform(images_t1ce)

            # 分割fake T1ce
            optimizer_seg_t.zero_grad()
            optimizer_g.zero_grad()

            fake_T1ce_1 = translator((images - 0.5) * 2.0)
            inputs = torch.cat([fake_T1ce_1, fake_T1ce_1], dim=1)
            outputs = segment_t((inputs + 1.0)*0.5)
            loss = 0.25 * BCELoss(outputs, masks) + 1.75 * FocalLoss(outputs, masks) + IoULoss(outputs, masks) + DiceLoss(outputs, masks)
            loss.backward()
            running_loss = running_loss + loss.item() * images.size(0)
            features_t.clear()

            optimizer_g.step()
            optimizer_seg_t.step()

            # 分割real T1
            optimizer_seg.zero_grad()
            optimizer_g.zero_grad()
            fake_T1ce = translator((images - 0.5)*2.0)

            with torch.no_grad():
                outputs_ = segment_t(torch.cat([(fake_T1ce.detach() + 1.0)*0.5, (fake_T1ce.detach() + 1.0)*0.5], dim=1))
                loss_ = 0.25 * BCELoss(outputs_, masks) + 1.75 * FocalLoss(outputs_, masks) + IoULoss(outputs_,masks) + DiceLoss(outputs_, masks)

            outputs_fake_T1ce = segment(torch.cat([images, (fake_T1ce + 1.0)*0.5], dim=1))
            # outputs_fake_T1ce = segment(torch.cat([images, images], dim=1))
            loss_fake_T1 = 0.25 * BCELoss(outputs_fake_T1ce, masks) + 1.75 * FocalLoss(outputs_fake_T1ce, masks) + IoULoss(outputs_fake_T1ce, masks) + DiceLoss(outputs_fake_T1ce, masks)

            if loss_fake_T1 >= loss_:
                loss_distill = utils.Distillation_loss(features_s, features_t)
            else:
                loss_distill = 0.0

            # loss_tumor_aware = L2loss(torch.mean(features_s[0], dim=1) * masks.detach(), torch.mean(features_t[0], dim=1) * masks.detach())

            # 清空特征
            features_t.clear()
            features_s.clear()

            loss_seg = loss_fake_T1 + loss_distill
            loss_seg.backward()
            running_loss_fake = running_loss_fake + loss_fake_T1.item() * images.size(0)

            optimizer_seg.step()
            optimizer_g.step()


            # 微调GAN
            # optimizer_g.zero_grad()
            # fake_T1ce_gan = translator((images - 0.5)*2.0)
            # instance_loss = L2loss(fake_T1ce_gan, (images_t1ce - 0.5)*2.0)
            # # pre_fake_B = discriminator_B(fake_T1ce_gan)
            # # loss_gan = BCELoss(pre_fake_B, torch.ones_like(pre_fake_B))
            #
            # loss_g = instance_loss
            # loss_g.backward()
            # running_loss_g = running_loss_g + loss_g.item() * images.size(0)
            # optimizer_g.step()

            # optimizer_d.zero_grad()
            # # Real loss
            # pred_real_B = discriminator_B((images_t1ce - 0.5)*2.0)
            # loss_D_real_B = BCELoss(pred_real_B, torch.ones_like(pred_real_B))
            #
            # # Fake loss
            # pred_fake_B = discriminator_B(translator((images - 0.5)*2.0).detach())
            # loss_D_fake_B = BCELoss(pred_fake_B, torch.zeros_like(pred_fake_B))
            #
            # # Total loss
            # loss_d_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            # loss_d = loss_d_B
            # loss_d.backward()
            # optimizer_d.step()

        # hook.remove()  # 移除钩子

        train_loss = running_loss / len(train_loader.dataset)
        train_loss_fake = running_loss_fake / len(train_loader.dataset)
        train_loss_g = running_loss_g / len(train_loader.dataset)
        train_loss_tal = running_loss_tal / len(train_loader.dataset)

        # 验证（只主进程打印/保存）
        segment.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for index, (images, masks, images_t1ce) in enumerate(val_loader):
                images = images.cuda()
                masks = masks.cuda()
                # images_t1ce = transform(images_t1ce.cuda())
                images_t1ce_fake = translator((images-0.5)*2.0)
                inputs = torch.cat([images, (images_t1ce_fake+1.0)*0.5], dim=1)
                # inputs = torch.cat([images, images], dim=1)
                outputs = segment(inputs)

                # 清空特征
                features_s.clear()

                loss = DiceLoss(outputs, masks)
                running_val_loss = running_val_loss + loss.item() * images.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)

        if is_main:
            print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | fake Loss: {train_loss_fake:.4f} | G-Loss: {train_loss_g:.4f} | TAL-Loss: {train_loss_tal:.4f}")
            # 保存模型
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(segment.module.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth".format(epoch)))
                torch.save(segment_t.module.state_dict(), os.path.join(args.checkpoint_dir, "best_t.pth".format(epoch)))
                torch.save(translator.module.state_dict(), os.path.join(args.checkpoint_dir, "best_G.pth".format(epoch)))
                print(f"Save Best Epoch!!!, Val Loss: {best_val_loss:.4f}")

            # torch.save(segment.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}.pth"))
            # torch.save(translator.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}" + '_G.pth'))

        # 更新学习率
        lr_scheduler_G.step()
        # lr_scheduler_D.step()
        lr_scheduler_seg.step()
        lr_scheduler_seg_t.step()
    print(best_val_loss)
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分布式训练U-Net用于语义分割")
    parser.add_argument("--train_image_dir", type=str, default='./data/train/T1_4')
    parser.add_argument("--train_image_t1ce_dir", type=str, default='./data/train/T1ce_4')
    parser.add_argument("--train_mask_dir", type=str, default='./data/train/seg_4')
    parser.add_argument("--val_image_dir", type=str, default='./data/test/T1_4')
    parser.add_argument("--val_image_t1ce_dir", type=str, default='./data/test/T1ce_4')
    parser.add_argument("--val_mask_dir", type=str, default='./data/test/seg_4')
    parser.add_argument("--trans_dir", type=str, default='./models/cyc_fddt4seg/fdit_netG_A2B.pth')
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/fddt4seg/fdit/")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
    args = parser.parse_args()

    train(args)
