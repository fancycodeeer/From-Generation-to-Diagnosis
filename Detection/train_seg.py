import argparse
import os
import torch
import warnings
torch.backends.cudnn.benchmark = True
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from datasets import SegDataset
import torch.nn.functional as F
from models import CenterNet
from models_t import Generator, Discriminator, FSeSimLoss
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
from utils import DiceLoss, FocalLoss, IoULoss, Distillation_loss
import torch.nn as nn
import gc

# ------------------ Seed Control ------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Paper hyperparameters
LR = 1e-4
BATCH_SIZE = 2
LR_STEP = [90, 120]
EPOCHS = 200
GAMMA = 0.1

def validate(epoch, model, translator, loader, device, rank):
    model.eval()
    translator.eval()
    val_dice_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for idx, (imgs, mask) in enumerate(loader):
            imgs = imgs.to(device)
            mask = mask.to(device)
            fake_mr = translator((imgs - 0.5) * 2)
            fake_mr = (fake_mr + 1) * 0.5
            mask_pred = model(torch.cat([imgs, fake_mr], dim=1))

            val_dice_loss = DiceLoss(mask_pred, mask).item() + val_dice_loss
            total_samples += imgs.size(0)

    # if rank == 0:
    #     print(
    #         f"Epoch {epoch} Val Dice Loss: {val_dice_loss / len(loader):.4f}")
    return val_dice_loss, total_samples


def main_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 钩子函数：提取中间层特征
    features_s = []
    features_t = []

    def register_hooks(model, hook_fn):
        # 遍历模型的所有层并为卷积层注册钩子
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                module.register_forward_hook(hook_fn)

    def hook_fn_s(module, input, output):
        features_s.append(output)

    def hook_fn_t(module, input, output):
        features_t.append(output.clone().detach())  # 将输出特征存储到列表中

    # dataset and loaders
    train_ds = SegDataset(args.train_img, args.train_ann)
    val_ds   = SegDataset(args.val_img,   args.val_ann)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, sampler=val_sampler,  num_workers=0)

    # model
    model = CenterNet(num_classes=1, n_channels=2).to(rank)
    model = DDP(model, device_ids=[rank])
    model_t = CenterNet(num_classes=1, n_channels=2).to(rank)
    model_t = DDP(model_t, device_ids=[rank])
    translator = Generator().to(rank)
    translator.load_state_dict(torch.load('./Translator/logs/best_G.pth'))
    translator = DDP(translator, device_ids=[rank])
    criterion_FSeSim = FSeSimLoss(device=rank, layers=('relu3_1', 'relu4_1'), patch_nums=8, patch_size=64, norm=True).to(rank)

    register_hooks(model, hook_fn=hook_fn_s)
    register_hooks(model_t, hook_fn=hook_fn_t)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizer_t = optim.Adam(model_t.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizer_trans = optim.Adam(translator.parameters(), lr=LR/10.0, betas=(0.9, 0.999))
    # lr scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEP, gamma=GAMMA)
    scheduler_t = optim.lr_scheduler.MultiStepLR(optimizer_t, milestones=LR_STEP, gamma=GAMMA)
    scheduler_trans = optim.lr_scheduler.MultiStepLR(optimizer_trans, milestones=LR_STEP, gamma=GAMMA)

    best_dice_loss = 100

    for epoch in range(1, EPOCHS+1):
        train_sampler.set_epoch(epoch)
        model.train()
        model_t.train()
        translator.train()
        Loss_t = 0.0
        Loss_s = 0.0

        for i, (imgs, mask) in enumerate(train_loader):
            imgs = imgs.to(rank)
            mask = mask.to(rank)

            optimizer_t.zero_grad()
            optimizer_trans.zero_grad()
            fake_mr = translator((imgs - 0.5) * 2)
            fake_mr = (fake_mr + 1) * 0.5

            mask_pred = model_t(torch.cat([fake_mr, fake_mr], dim=1))
            # focal loss (BCE)
            loss_t = 0.25 * F.binary_cross_entropy(mask_pred, mask) + 1.75 * FocalLoss(mask_pred, mask) + DiceLoss(
                mask_pred, mask) + IoULoss(mask_pred, mask)

            # combined
            loss_t.backward()
            Loss_t += loss_t.item()
            optimizer_t.step()
            optimizer_trans.step()

            optimizer.zero_grad()
            optimizer_trans.zero_grad()
            fake_mr = translator((imgs - 0.5) * 2)
            fake_mr = (fake_mr + 1) * 0.5
            mask_pred = model(torch.cat([imgs, fake_mr], dim=1))
            # focal loss (BCE) for heatmap
            loss_s = 0.25 * F.binary_cross_entropy(mask_pred, mask) + 1.75 * FocalLoss(mask_pred, mask) + DiceLoss(
                mask_pred, mask) + IoULoss(mask_pred, mask)

            if loss_s >= loss_t:
                loss_distill = Distillation_loss(features_s, features_t)
            else:
                loss_distill = 0.0

            loss = loss_s + loss_distill
            loss.backward()

            optimizer.step()
            optimizer_trans.step()
            Loss_s += loss_s.item()

            features_t.clear()
            features_s.clear()


            optimizer_trans.zero_grad()
            fake_MR = translator((imgs - 0.5) * 2)
            fake_MR = (fake_MR + 1) * 0.5
            # FSeSim 结构一致性损失
            loss_S = FocalLoss(fake_MR, mask) * 0.1
            loss_S.backward()
            optimizer_trans.step()

        if rank == 0:
            print(f"Epoch {epoch} loss_t: {Loss_t / len(train_loader):.4f} | loss_s: {Loss_s / len(train_loader):.4f}")

        scheduler.step()
        scheduler_t.step()
        scheduler_trans.step()
        gc.collect()
        torch.cuda.empty_cache()


        current_dice, total_samples = validate(epoch, model, translator, val_loader, rank, rank)
        features_t.clear()
        features_s.clear()

        # 转换为Tensor用于分布式通信
        val_loss_tensor = torch.tensor(current_dice).float().to(rank)
        val_num_tensor = torch.tensor(total_samples).float().to(rank)
        # 汇总所有GPU的数据
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_num_tensor, op=dist.ReduceOp.SUM)

        # 计算全局平均损失
        global_val_loss = val_loss_tensor.item() / val_num_tensor.item()

        # save best
        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(args.save_dir, '{}_seg.pth'.format(epoch)))

        if rank == 0 and global_val_loss < best_dice_loss:
            best_dice_loss = global_val_loss
            torch.save(model.module.state_dict(), os.path.join(args.save_dir, 'best_dice.pth'))
            torch.save(model_t.module.state_dict(), os.path.join(args.save_dir, '{}_t.pth'.format(epoch)))
            torch.save(translator.module.state_dict(), os.path.join(args.save_dir, '{}_g.pth'.format(epoch)))

            print(f"best dice_loss: {best_dice_loss:.4f}")
    dist.destroy_process_group()



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_img', default='./data4seg/train/img')
    p.add_argument('--train_ann', default='./data4seg/train/maks')
    p.add_argument('--val_img',   default='./data4seg/val/img')
    p.add_argument('--val_ann',   default='./data4seg/val/maks')
    p.add_argument('--save_dir',  default='./logs/seg_dist')
    p.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    world_size = args.world_size
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
