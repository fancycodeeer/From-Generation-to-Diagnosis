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
from utils import decode, top1_peak_coords
import torch.nn.functional as F
from models import CenterNet
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

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


def train_one_epoch(epoch, model, loader, optimizer, device, rank):
    model.train()
    hm_loss = 0.0
    wh_loss = 0.0
    off_loss = 0.0
    for i, (imgs, tgt) in enumerate(loader):
        imgs = imgs.to(device)
        hm_gt = torch.stack([t['heatmap'] for t in tgt]).to(device)
        # wh_gt = torch.stack([t['wh'] for t in tgt]).to(device)
        # off_gt = torch.stack([t['reg'] for t in tgt]).to(device)
        # mask = torch.stack([t['reg_mask'] for t in tgt]).to(device)

        hm_pred = model(imgs)
        # print(wh_gt, off_gt)
        # print(hm_pred.min(), hm_pred.max())

        # print(hm_gt.min(), hm_gt.max())
        # print(mask.sum())

        # focal loss (BCE) for heatmap
        loss_hm = F.binary_cross_entropy(hm_pred, hm_gt) + F.mse_loss(hm_pred, hm_gt)
        # size loss
        # B, C, H, W = wh_pred.shape
        # wh_pred_flat = wh_pred.permute(0,2,3,1).reshape(-1,2)
        # wh_gt_flat   = wh_gt.permute(0,2,3,1).reshape(-1,2)
        # pos = mask.view(-1) > 0
        # loss_wh = F.l1_loss(wh_pred_flat[pos], wh_gt_flat[pos])

        # offset loss
        # off_pred_flat = off_pred.permute(0,2,3,1).reshape(-1,2)
        # off_gt_flat   = off_gt.permute(0,2,3,1).reshape(-1,2)
        # loss_off = F.l1_loss(off_pred_flat[pos], off_gt_flat[pos])

        # combined
        loss = loss_hm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        hm_loss += loss_hm.item()
        # off_loss += loss_off.item()
    if rank == 0:
        print(f"Epoch {epoch} hm loss: {hm_loss/len(loader):.4f} ")


def validate(epoch, model, loader, device, rank):
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


            if rank == 0 and idx == 0:
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
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # dataset and loaders
    train_ds = CenterNetDataset(args.train_img, args.train_ann)
    val_ds   = CenterNetDataset(args.val_img,   args.val_ann)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, collate_fn=detection_collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=1, sampler=val_sampler,   num_workers=4, collate_fn=detection_collate_fn)

    # model
    model = CenterNet(num_classes=1).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    # lr scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEP, gamma=GAMMA)

    best_error = 1e8
    best_kl = 1e8

    for epoch in range(1, EPOCHS+1):
        train_sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, train_loader, optimizer, rank, rank)
        scheduler.step()
        current_error, current_kl = validate(epoch, model, val_loader, rank, rank)
        # save best
        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(args.save_dir, '{}_detect.pth'.format(epoch)))

        if rank == 0 and current_error < best_error:
            best_error = current_error
            torch.save(model.module.state_dict(), os.path.join(args.save_dir, 'best_error.pth'))
        if rank == 0 and current_kl < best_kl:
            best_kl = current_kl
            torch.save(model.module.state_dict(), os.path.join(args.save_dir, 'best_kl.pth'))

            print(f"best error: {best_error:.4f} | best kl: {best_kl:.4f}")
    dist.destroy_process_group()



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_img', default='./data/train/images')
    p.add_argument('--train_ann', default='./data/train/annotations')
    p.add_argument('--val_img',   default='./data/val/images')
    p.add_argument('--val_ann',   default='./data/val/annotations')
    p.add_argument('--save_dir',  default='./logs/detect')
    p.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    world_size = args.world_size
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
