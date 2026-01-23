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
from models import UNet, Generator, Discriminator, ResNet50
import torchvision.transforms as transforms
import utils
import random
import numpy as np
import itertools
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from utils.utils import weights_init
from utils import Fourier_trans, make_mask_, make_distance_mask_
from utils.loss import PixelSelfAttention, js_divergence


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
    # transform_train = transforms.Compose([
    #     transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
    #     transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    #     transforms.Grayscale()
    # ])
    transform = transforms.Compose([transforms.Grayscale()])

    # 数据集 & 分布式采样器
    train_dataset = SegmentationDataset(args.train_image_dir, args.train_mask_dir, args.train_image_t1ce_dir, transform)
    val_dataset = SegmentationDataset(args.val_image_dir, args.val_mask_dir, args.val_image_t1ce_dir, transform)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=8, pin_memory=True, drop_last=True)

    # 模型 & DDP封装
    segment = UNet(n_channels=2, n_classes=1, bilinear=True).cuda()
    segment = DDP(segment, device_ids=[device])
    segment_t = UNet(n_channels=2, n_classes=1, bilinear=True).cuda()
    segment_t = DDP(segment_t, device_ids=[device])

    translator = Generator(1, 1).cuda()
    translator.load_state_dict(torch.load('./models/299_netG_A2B.pth'))
    translator = DDP(translator, device_ids=[device])

    # psa = PixelSelfAttention(in_channels=1, embed_dim=3).cuda()
    # psa = DDP(psa, device_ids=[device])
    # psa_ = PixelSelfAttention(in_channels=1, proj=False).cuda()

    cls = ResNet50(num_classes=2, dim=3).cuda()
    cls = DDP(cls, device_ids=[device])
    cls_t = ResNet50(num_classes=2, dim=3).cuda()
    cls_t = DDP(cls_t, device_ids=[device])



    # 钩子函数：提取中间层特征
    features_s = []
    features_t = []
    cls_feature_s = []
    cls_feature_t = []

    def register_hooks_cls(model, hook_fn):
        # 遍历模型的所有层并为卷积层注册钩子
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(hook_fn)

    def register_hooks_seg(model, hook_fn):
        # 遍历模型的所有层并为卷积层注册钩子
        # for name, module in model.named_modules():
        #     if name.endswith('down4'):
        #         module.register_forward_hook(hook_fn)
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(hook_fn)


    def hook_fn_s(module, input, output):
        features_s.append(output)

    def hook_fn_t(module, input, output):
        features_t.append(output)  # 将输出特征存储到列表中

    def hook_fn_cls_s(module, input, output):
        cls_feature_s.append(output)

    def hook_fn_cls_t(module, input, output):
        cls_feature_t.append(output)  # 将输出特征存储到列表中


    # 注册钩子
    register_hooks_cls(cls, hook_fn=hook_fn_cls_s)
    register_hooks_cls(cls_t, hook_fn=hook_fn_cls_t)
    register_hooks_seg(segment, hook_fn=hook_fn_s)
    register_hooks_seg(segment_t, hook_fn=hook_fn_t)

    # 损失函数
    BCELoss = nn.BCEWithLogitsLoss()
    CELoss = nn.CrossEntropyLoss()
    DiceLoss = utils.DiceLoss()
    FocalLoss = utils.FocalLoss(alpha=0.25, gamma=5, reduction='mean')
    IoULoss = utils.IoULoss()
    L2loss = torch.nn.MSELoss()

    # 设置优化器
    optimizer_seg = optim.Adam(itertools.chain(segment.parameters(), cls.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optimizer_seg_t = optim.Adam(itertools.chain(segment_t.parameters(), cls_t.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(translator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=utils.LambdaLR(args.epochs, 0, 30).step)
    lr_scheduler_seg = torch.optim.lr_scheduler.LambdaLR(optimizer_seg, lr_lambda=utils.LambdaLR(args.epochs, 0, 30).step)
    lr_scheduler_seg_t = torch.optim.lr_scheduler.LambdaLR(optimizer_seg_t, lr_lambda=utils.LambdaLR(args.epochs, 0, 30).step)
    best_val_loss = float('inf')
    best_f1 = 0
    best_acc = 0
    best_recall = 0
    best_precision = 0

    for epoch in range(args.epochs):
        segment.train()
        segment_t.train()
        translator.train()
        cls.train()
        cls_t.train()

        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        running_loss_s = 0.0
        running_loss_g = 0.0

        for index, (images, masks, images_t1ce) in enumerate(train_loader):
            images = images.cuda()
            masks = masks.cuda()
            images_t1ce = images_t1ce.cuda()
            images_t1ce = transform(images_t1ce)

            # if is_main:
            #     fake_T1_xx = translator(images_t1ce)
            #     fake_T1_xx = 0.5 * (fake_T1_xx.data + 1.0)
            #     save_image(fake_T1_xx, './output/trans/' + str(index) + '.png')

            # 分割teacher real T1
            optimizer_seg_t.zero_grad()
            optimizer_g.zero_grad()
            optimizer_seg.zero_grad()

            fake_T1ce_1 = translator((images - 0.5) * 2.0)
            inputs = torch.cat([fake_T1ce_1, fake_T1ce_1], dim=1)
            # inputs = torch.cat([images_t1ce, images_t1ce], dim=1)
            outputs = segment_t(inputs)
            loss = 0.25 * BCELoss(outputs, masks) + 1.75 * FocalLoss(outputs, masks) + IoULoss(outputs, masks) + DiceLoss(outputs, masks)
            # loss_attn = js_divergence(torch.flatten(psa_(masks), 1), torch.flatten(psa_(fake_T1ce_1), 1))

            zero_mask_t = (masks.view(args.batch_size, -1).sum(dim=1) == 0)  # shape (B,)
            labels_t = (~zero_mask_t).long()  # (B,)
            pred_t = cls_t(torch.cat([inputs, outputs], dim=1))
            loss_t_cls = CELoss(pred_t, labels_t)

            loss_t = loss + loss_t_cls
            # loss_t.backward()
            running_loss = running_loss + loss_t_cls.item() * images.size(0)


            # student
            # 找出 batch 中哪些样本 mask 全零
            # zero_mask: True 表示该样本是全零 mask
            zero_mask = (masks.view(args.batch_size, -1).sum(dim=1) == 0)  # shape [B]
            non_zero = ~zero_mask  # shape [B]

            fake_T1ce = translator((images - 0.5) * 2.0)
            inputs_s = torch.cat([images, (fake_T1ce + 1.0) * 0.5], dim=1)
            # inputs_s = torch.cat([images, images_t1ce], dim=1)
            outputs_s = segment(inputs_s)

            # # 2. 只对非空样本计算分割损失
            # if non_zero.sum() > 0:
            #     outputs_nz = outputs_s[non_zero]  # 选出对应输出
            #     masks_nz = masks[non_zero]  # 选出对应 mask
            #     loss_s_seg = (
            #             0.25 * BCELoss(outputs_nz, masks_nz)
            #             + 1.75 * FocalLoss(outputs_nz, masks_nz)
            #             + IoULoss(outputs_nz, masks_nz)
            #             + DiceLoss(outputs_nz, masks_nz)
            #     )
            # else:
            #     # 如果整个 batch 都是空 mask，就把 seg loss 设为 0
            #     loss_s_seg = torch.tensor(0.0, device=masks.device)
            loss_s_seg = (
                    0.25 * BCELoss(outputs_s, masks)
                    + 1.75 * FocalLoss(outputs_s, masks)
                    + IoULoss(outputs_s, masks)
                    + DiceLoss(outputs_s, masks)
            )

            # 分类分支
            labels = (~zero_mask).long()       # (B,)
            pred_s = cls(torch.cat([inputs_s, outputs_s], dim=1))
            loss_s_cls = CELoss(pred_s, labels)

            if loss_s_cls >= loss_t_cls:
                loss_distill = utils.Distillation_loss(cls_feature_s, cls_feature_t)
            else:
                loss_distill = 0.0

            if loss_s_seg >= loss:
                loss_distill = loss_distill + utils.Distillation_loss(features_s, features_t)
            else:
                loss_distill = loss_distill + 0.0

            loss_s = loss_s_cls + loss_s_seg + loss_distill
            loss_total = loss_s + loss_t
            loss_total.backward()
            running_loss_s = running_loss_s + loss_s_cls.item() * images.size(0)

            # 清空特征
            features_t.clear()
            features_s.clear()
            cls_feature_t.clear()
            cls_feature_s.clear()

            optimizer_seg_t.step()
            optimizer_seg.step()
            optimizer_g.step()

            # ----微调GAN
            optimizer_g.zero_grad()
            fake_T1ce_gan = translator((images - 0.5)*2.0)
            instance_loss = L2loss(fake_T1ce_gan, (images_t1ce - 0.5)*2.0)
            loss_g = instance_loss * 5
            loss_g.backward()
            running_loss_g = running_loss_g + loss_g.item() * images.size(0)
            optimizer_g.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_loss_s = running_loss_s / len(train_loader.dataset)
        train_loss_g = running_loss_g / len(train_loader.dataset)

        # 验证（只主进程打印/保存）
        segment.eval()
        cls.eval()
        running_val_loss = 0.0
        # with torch.no_grad():
        #     for index, (images, masks, images_t1ce) in enumerate(val_loader):
        #         images = images.cuda()
        #         masks = masks.cuda()
        #         # images_t1ce = transform(images_t1ce.cuda())
        #         # images_t1ce = (translator((images-0.5)*2.0) + 1.0) * 0.5
        #
        #         inputs = torch.cat([images, images], dim=1)
        #         outputs = segment(inputs)
        #         zero_mask = (masks.view(args.batch_size, -1).sum(dim=1) == 0)  # shape (B,)
        #         labels = (~zero_mask).long()      # (B,)
        #         loss_s_cls = CELoss(cls(torch.cat([images, images, outputs], dim=1)), labels)
        #         # loss = DiceLoss(outputs, masks)
        #         features_s.clear()
        #         cls_feature_s.clear()
        #         running_val_loss = running_val_loss + loss_s_cls.item() * images.size(0)

        # 累计指标
        tp = torch.tensor(0, dtype=torch.long, device=device)
        tn = torch.tensor(0, dtype=torch.long, device=device)
        fp = torch.tensor(0, dtype=torch.long, device=device)
        fn = torch.tensor(0, dtype=torch.long, device=device)

        with torch.no_grad():
            for index, (images, masks, images_t1ce) in enumerate(val_loader):
                images = images.cuda()
                masks = masks.cuda()
                # images_t1ce = transform(images_t1ce.cuda())
                images_t1ce = (translator((images-0.5)*2.0) + 1.0) * 0.5

                inputs = torch.cat([images, images_t1ce], dim=1)
                outputs = segment(inputs)
                logits = cls(torch.cat([inputs, outputs], dim=1))
                preds = torch.argmax(logits, dim=1)

                # 清空特征
                features_s.clear()
                cls_feature_s.clear()

                zero_mask = (masks.view(args.batch_size, -1).sum(dim=1) == 0)  # shape (B,)
                labels = (~zero_mask).long()  # (B,)
                if is_main:
                    print('label:', labels)

                tp += ((preds == 1) & (labels == 1)).sum()
                tn += ((preds == 0) & (labels == 0)).sum()
                fp += ((preds == 1) & (labels == 0)).sum()
                fn += ((preds == 0) & (labels == 1)).sum()

        # 跨卡汇总
        dist.all_reduce(tp, op=dist.ReduceOp.SUM)
        dist.all_reduce(tn, op=dist.ReduceOp.SUM)
        dist.all_reduce(fp, op=dist.ReduceOp.SUM)
        dist.all_reduce(fn, op=dist.ReduceOp.SUM)

        # 仅 rank 0 计算并打印
        if dist.get_rank() == 0:
            tp_, tn_, fp_, fn_ = tp.item(), tn.item(), fp.item(), fn.item()
            total = tp_ + tn_ + fp_ + fn_
            acc = (tp_ + tn_) / total if total > 0 else 0
            prec = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
            rec = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"[Epoch {epoch:03d}]  Classification: "f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        val_loss = running_val_loss / len(val_loader.dataset)

        if is_main:
            print(f"Epoch [{epoch}/{args.epochs}] | Teacher Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Student Loss: {train_loss_s:.4f} | G-Loss: {train_loss_g:.4f}")
            # 保存模型
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_recall = rec
                best_precision = prec
                torch.save(segment.module.state_dict(), os.path.join(args.checkpoint_dir, "best_model_{}.pth".format(epoch)))
                print(f"Save Best Epoch!!!, Val Loss: {best_f1:.4f}")

            torch.save(segment.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}.pth"))
            torch.save(cls.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}" + '_cls.pth'))
            torch.save(segment_t.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}" + '_t.pth'))
            torch.save(cls_t.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}" + '_cls_t.pth'))
            # torch.save(psa.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}" + '_psa.pth'))
            torch.save(translator.module.state_dict(), os.path.join(args.checkpoint_dir, f"{epoch}" + '_G.pth'))


        # 更新学习率
        lr_scheduler_G.step()
        lr_scheduler_seg.step()
        lr_scheduler_seg_t.step()

    print(f"Classification: "f"Acc: {best_acc:.4f}, Prec: {best_precision:.4f}, Rec: {best_recall:.4f}, F1: {best_f1:.4f}")
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分布式训练U-Net用于语义分割")
    parser.add_argument("--train_image_dir", type=str, default='./data/classification/seg/train/T1')
    parser.add_argument("--train_image_t1ce_dir", type=str, default='./data/classification/seg/train/T1ce')
    parser.add_argument("--train_mask_dir", type=str, default='./data/classification/seg/train/seg')

    parser.add_argument("--val_image_dir", type=str, default='./data/classification/seg/test/T1')
    parser.add_argument("--val_image_t1ce_dir", type=str, default='./data/classification/seg/test/T1ce')
    parser.add_argument("--val_mask_dir", type=str, default='./data/classification/seg/test/seg')

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/seg+cls/T1_distill_full_cls_jilian")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    args = parser.parse_args()

    train(args)
