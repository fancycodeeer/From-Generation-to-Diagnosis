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
from datasets import CenterNetDataset, detection_collate_fn
from utils import decode, top1_peak_coords, Distillation_loss
import torch.nn.functional as F
from models import CenterNet
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import torch.nn as nn
from models_t import Generator, Discriminator, FSeSimLoss
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
EPOCHS = 140
GAMMA = 0.1
WEIGHT_WH = 0.1
WEIGHT_OFF = 1.0

def validate(epoch, model, loader, device, rank, plt=True):
    model.eval()
    val_hm_loss = 0.0
    total_kl = 0.0
    total_err = 0.0

    with torch.no_grad():
        for idx, (imgs, tgt) in enumerate(loader):
            imgs = imgs.to(device)
            hm_pred = model(imgs)
            hm_gt = torch.stack([t['heatmap'] for t in tgt]).to(device)

            B, C, H, W = imgs.shape

            # wh_gt = torch.stack([t['wh'] for t in tgt]).to(device)
            # mask = torch.stack([t['reg_mask'] for t in tgt]).to(device)

            val_hm_loss = F.binary_cross_entropy(hm_pred, hm_gt).item() + val_hm_loss

            # KL divergence per sample
            pred_flat = hm_pred.view(B, -1)
            gt_flat = hm_gt.view(B, -1)
            pred_dist = pred_flat / (pred_flat.sum(dim=1, keepdim=True) + 1e-6)
            gt_dist = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-6)
            # compute KL per sample
            kl_per_sample = F.kl_div(pred_dist.log(), gt_dist, reduction='batchmean')
            total_kl += kl_per_sample.item()

            # Euclidean distance of top1 peaks per channel
            coords_pred, _ = top1_peak_coords(hm_pred)
            coords_gt, _ = top1_peak_coords(hm_gt)
            diff = coords_pred.float() - coords_gt.float()  # (B, C, 2)
            dists = torch.norm(diff, dim=2)  # (B, C)
            total_err += dists.mean()


            if plt and rank == 0 and idx == 0:
                # 取第0张图的 GT & Pred
                gt = hm_gt[0, 0].cpu().detach().numpy()  # [fh,fw]
                pr = hm_pred[0, 0].cpu().detach().numpy()  # [fh,fw]
                # 并排可视化
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.title("GT Heatmap")
                plt.imshow(gt, cmap='jet')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.title("Pred Heatmap")
                plt.imshow(pr, cmap='jet')
                plt.colorbar(fraction=0.046, pad=0.04)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                if idx == 0:
                    plt.close()

    val_hm_loss = val_hm_loss / len(loader)
    avg_kl = total_kl / len(loader)
    avg_err = total_err / len(loader)

    if rank == 0:
        print(
            f"Epoch {epoch} Val HM Loss: {val_hm_loss:.4f} | Mean Error: {avg_err:.4f} | KL Divergence: {avg_kl:.4f}")
    return avg_err, avg_kl


def main_worker(rank, world_size, args):
    torch.autograd.set_detect_anomaly(True)
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
        features_t.append(output.detach())  # 将输出特征存储到列表中

    # dataset and loaders
    train_ds = CenterNetDataset(args.train_img, args.train_ann)
    val_ds   = CenterNetDataset(args.val_img,   args.val_ann)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=1, collate_fn=detection_collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=1, sampler=val_sampler,   num_workers=1, collate_fn=detection_collate_fn)

    # model
    model_s = CenterNet(num_classes=1, n_channels=2).to(rank)
    model_s = DDP(model_s, device_ids=[rank])
    model_t = CenterNet(num_classes=1, n_channels=2).to(rank)
    model_t = DDP(model_t, device_ids=[rank])
    translator = Generator().to(rank)
    translator.load_state_dict(torch.load('./Translator/logs/best_G.pth'))
    translator = DDP(translator, device_ids=[rank])
    criterion_FSeSim = FSeSimLoss(device=rank, layers=('relu3_1', 'relu4_1'), patch_nums=8, patch_size=64, norm=True).to(rank)


    register_hooks(model_s, hook_fn=hook_fn_s)
    register_hooks(model_t, hook_fn=hook_fn_t)

    optimizer_s = optim.Adam(model_s.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizer_t = optim.Adam(model_t.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizer_trans = optim.Adam(translator.parameters(), lr=LR, betas=(0.9, 0.999))

    # lr scheduler
    scheduler_s = optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=LR_STEP, gamma=GAMMA)
    scheduler_t = optim.lr_scheduler.MultiStepLR(optimizer_t, milestones=LR_STEP, gamma=GAMMA)
    scheduler_trans = optim.lr_scheduler.MultiStepLR(optimizer_trans, milestones=LR_STEP, gamma=GAMMA)

    best_error = 1e8
    best_kl = 1e8

    for epoch in range(1, EPOCHS+1):
        train_sampler.set_epoch(epoch)

        model_s.train()
        model_t.train()
        translator.train()
        hm_loss = 0.0

        for i, (imgs, tgt) in enumerate(train_loader):
            imgs = imgs.to(rank)
            hm_gt = torch.stack([t['heatmap'] for t in tgt]).to(rank)

            optimizer_t.zero_grad()
            optimizer_trans.zero_grad()
            fake_mr = translator((imgs - 0.5) * 2)
            fake_mr = (fake_mr + 1) * 0.5
            hm_pred_t = model_t(torch.cat([fake_mr, fake_mr], dim=1))
            # focal loss (BCE) for heatmap
            loss_hm_t = F.binary_cross_entropy(hm_pred_t, hm_gt) + F.mse_loss(hm_pred_t, hm_gt)
            loss_hm_t.backward()
            optimizer_trans.step()
            optimizer_t.step()

            optimizer_s.zero_grad()
            optimizer_trans.zero_grad()
            fake_mr = translator((imgs - 0.5) * 2)
            fake_mr = (fake_mr + 1) * 0.5
            hm_pred = model_s(torch.cat([imgs, fake_mr], dim=1))
            # focal loss (BCE) for heatmap
            loss_hm = F.binary_cross_entropy(hm_pred, hm_gt) + F.mse_loss(hm_pred, hm_gt)

            # if loss_hm >= loss_hm_t:
            #     loss_distill = Distillation_loss(features_s, features_t)
            # else:
            #     loss_distill = 0.0

            loss_distill = Distillation_loss(features_s, features_t)

            loss = loss_hm + loss_distill
            loss.backward()
            optimizer_trans.step()
            optimizer_s.step()

            # optimizer_trans.zero_grad()
            # fake_MR = translator((imgs - 0.5) * 2)
            # fake_MR = (fake_MR + 1) * 0.5
            # # FSeSim 结构一致性损失
            # loss_S = criterion_FSeSim(imgs, fake_MR) * 0.5
            # loss_S.backward()
            # optimizer_trans.step()


            hm_loss += loss_hm.item()
            # off_loss += loss_off.item()
            features_t.clear()
            features_s.clear()
        if rank == 0:
            print(f"Epoch {epoch} hm loss: {hm_loss / len(train_loader):.4f} ")

        scheduler_s.step()
        scheduler_t.step()
        scheduler_trans.step()
        gc.collect()
        torch.cuda.empty_cache()


        # current_error, current_kl = validate(epoch, model_s, val_loader, rank, rank,plt=False)
        # features_t.clear()
        # features_s.clear()

        # save best
        if rank == 0:
            torch.save(model_s.module.state_dict(), os.path.join(args.save_dir, '{}_s.pth'.format(epoch)))
            torch.save(model_t.module.state_dict(), os.path.join(args.save_dir, '{}_t.pth'.format(epoch)))
            torch.save(translator.module.state_dict(), os.path.join(args.save_dir, '{}_g.pth'.format(epoch)))

        # if rank == 0 and current_error < best_error:
        #     best_error = current_error
        #     torch.save(model_s.module.state_dict(), os.path.join(args.save_dir, 'best_error.pth'))
        # if rank == 0 and current_kl < best_kl:
        #     best_kl = current_kl
        #     torch.save(model_s.module.state_dict(), os.path.join(args.save_dir, 'best_kl.pth'))
        #
        #     print(f"best error: {best_error:.4f} | best kl: {best_kl:.4f}")

        torch.cuda.ipc_collect()
    dist.destroy_process_group()



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_img', default='./data/train/images')
    p.add_argument('--train_ann', default='./data/train/annotations')
    p.add_argument('--val_img',   default='./data/val/images')
    p.add_argument('--val_ann',   default='./data/val/annotations')
    p.add_argument('--save_dir',  default='./logs/detect_dist_fsesim_ablation')
    p.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    world_size = args.world_size
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
