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
from dataset import ClassificationDataset
from models import UNet, Generator, Discriminator, ResNet50, ResNet18, convnext_tiny

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
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        # transforms.RandomRotation(30),  # 随机旋转，旋转角度范围是 [-30, 30]
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整亮度、对比度、饱和度和色调
        transforms.RandomVerticalFlip(), # 随机垂直翻转
        ]
        )


    # 数据集 & 分布式采样器
    train_dataset = ClassificationDataset(args.train_image_0_dir, args.train_image_1_dir, args.train_image_t1ce_0_dir, args.train_image_t1ce_1_dir, transform)
    val_dataset = ClassificationDataset(args.val_image_0_dir, args.val_image_1_dir, args.val_image_t1ce_0_dir, args.val_image_t1ce_1_dir, transforms.Grayscale())

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8, pin_memory=True, drop_last=True)

    # 模型 & DDP封装
    model = ResNet18(num_classes=2, dim=1).cuda()
    model = DDP(model, device_ids=[device])
    model_t = ResNet18(num_classes=2, dim=1).cuda()
    model_t = DDP(model_t, device_ids=[device])
    translator = Generator(1, 1).cuda()
    translator.load_state_dict(torch.load('./models/299_netG_A2B.pth',  weights_only=True))
    translator = DDP(translator, device_ids=[device])

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
        features_t.append(output)  # 将输出特征存储到列表中

    # def hook_fn_trans_s(module, input, output):
    #     features_trans_s.append(output)  # 将输出特征存储到列表中
    # def hook_fn_trans_t(module, input, output):
    #     features_trans_t.append(output)  # 将输出特征存储到列表中

    # 注册钩子
    register_hooks(model, hook_fn=hook_fn_s)
    register_hooks(model_t, hook_fn=hook_fn_t)
    # register_hooks(translator, hook_fn=hook_fn_trans_s)
    # register_hooks(translator_t, hook_fn=hook_fn_trans_t)

    # 损失函数
    CELoss = nn.CrossEntropyLoss()

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_t = optim.Adam(model_t.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optimizer_g = optim.Adam(translator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=utils.LambdaLR(args.epochs, 0, 50).step)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, 50).step)
    lr_scheduler_t = torch.optim.lr_scheduler.LambdaLR(optimizer_t, lr_lambda=utils.LambdaLR(args.epochs, 0, 50).step)
    best_val_loss = float('inf')

    label_0 = torch.tensor([0]).cuda().repeat(args.batch_size)
    label_1 = torch.tensor([1]).cuda().repeat(args.batch_size)

    best_val_acc = 0

    for epoch in range(args.epochs):
        model.train()
        translator.train()
        model_t.train()

        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        running_loss_fake = 0.0
        running_loss_g = 0.0

        correct_train = 0
        total_train = 0

        for index, (t1_1_image, t1ce_1_image, t1_2_image, t1ce_2_image) in enumerate(train_loader):
            input_1 = torch.cat([t1_1_image, t1ce_1_image], dim=1).cuda()
            input_2 = torch.cat([t1_2_image, t1ce_2_image], dim=1).cuda()

            #
            optimizer.zero_grad()
            outputs = model(input_1)

            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == label_0).sum().item()
            total_train += label_0.size(0)

            loss_1 = CELoss(outputs, label_0)
            loss_1.backward()
            optimizer.step()

            optimizer.zero_grad()
            outputs_1 = model(input_2)

            _, predicted = torch.max(outputs_1, 1)
            correct_train += (predicted == label_1).sum().item()
            total_train += label_1.size(0)

            loss_2 = CELoss(outputs_1, label_1)
            loss_2.backward()

            loss = loss_1 + loss_2
            running_loss = running_loss + loss.item() * t1_1_image.size(0)

            # 清空特征
            features_t.clear()
            features_s.clear()

            optimizer.step()



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

        # hook.remove()  # 移除钩子

        train_loss = running_loss / len(train_loader.dataset)
        train_loss_fake = running_loss_fake / len(train_loader.dataset)
        train_loss_g = running_loss_g / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # 验证（只主进程打印/保存）
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for index, (t1_1_image, t1ce_1_image, t1_2_image, t1ce_2_image) in enumerate(val_loader):
                # t1ce_1_image = t1ce_1_image.cuda()
                # t1ce_2_image = t1ce_2_image.cuda()
                input_1 = torch.cat([t1_1_image, t1ce_1_image], dim=1).cuda()
                input_2 = torch.cat([t1_2_image, t1ce_2_image], dim=1).cuda()

                outputs = model(input_1)
                outputs_1 = model(input_2)
                # 清空特征
                features_s.clear()
                loss = CELoss(outputs, label_0) + CELoss(outputs_1, label_1)
                running_val_loss = running_val_loss + loss.item() * t1_1_image.size(0)

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == label_0).sum().item()
                total_val += label_0.size(0)
                _, predicted_1 = torch.max(outputs_1, 1)
                correct_val += (predicted_1 == label_1).sum().item()
                total_val += label_1.size(0)


        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val

        if is_main:
            print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            # 保存模型
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                # torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, "best_model_{}.pth".format(epoch)))
                torch.save(model.module.state_dict(),os.path.join(args.checkpoint_dir, "best_model.pth"))
                print(f"Save Best Epoch!!!, Val Acc: {best_val_acc:.4f}")

            torch.save(model.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}.pth"))
            # torch.save(translator.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}" + '_G.pth'))

        # 更新学习率
        # lr_scheduler_G.step()
        lr_scheduler.step()
        # lr_scheduler_t.step()
    print(best_epoch)
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分布式训练U-Net用于语义分割")
    parser.add_argument("--train_image_0_dir", type=str, default='./data/classification/train/T1_0')
    parser.add_argument("--train_image_t1ce_0_dir", type=str, default='./data/classification/train/T1ce_0')
    parser.add_argument("--train_image_1_dir", type=str, default='./data/classification/train/T1_4')
    parser.add_argument("--train_image_t1ce_1_dir", type=str, default='./data/classification/train/T1ce_4')

    parser.add_argument("--val_image_0_dir", type=str, default='./data/classification/val/T1_0')
    parser.add_argument("--val_image_t1ce_0_dir", type=str, default='./data/classification/val/T1ce_0')
    parser.add_argument("--val_image_1_dir", type=str, default='./data/classification/val/T1_4')
    parser.add_argument("--val_image_t1ce_1_dir", type=str, default='./data/classification/val/T1ce_4')

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/classify_mix")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    args = parser.parse_args()

    train(args)
