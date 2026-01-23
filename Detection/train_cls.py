import argparse
import os
import torch
import warnings
torch.backends.cudnn.benchmark = True
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from classifier import ResNet, ResNet18Backbone
import torch.optim as optim
from datasets import ClassifyDataset, ClassifyDatasetStroke
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
import itertools


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

def validate(
             classifier,
             loader,
             device,
             rank,
             thresh: float = 0.5,
):
    """
    计算 val_loader 上的 acc / precision / recall / f1.

    Args
    ----
    model_s    : CenterNet (DDP 包装或裸模块均可)
    classifier : ResNet50 (DDP or nn.Module)
    loader     : DataLoader
    device     : 当前 rank 对应的 cuda 设备
    rank       : 当前进程 rank
    thresh     : 判定为正样本的阈值, 默认 0.5
    """
    classifier.eval()

    tp = fp = tn = fn = 0
    print("length:{}".format(len(loader)))

    for imgs, _, label in loader:           # label shape: (B, 1) or (B,)
        imgs   = imgs.to(device)
        label  = label.to(device).float()       # 0/1
        # hm_pred = model_s(torch.cat([imgs, imgs], dim=1))
        prob    = classifier(torch.cat([imgs, imgs, imgs], dim=1)).squeeze(1)  # (B,)
        # if rank == 0:
        #     print(prob, label)

        pred = (prob >= thresh).float()         # 0 / 1

        tp += ((pred == 1) & (label == 1)).sum().item()
        fp += ((pred == 1) & (label == 0)).sum().item()
        tn += ((pred == 0) & (label == 0)).sum().item()
        fn += ((pred == 0) & (label == 1)).sum().item()

    # ---- DDP: 聚合四个计数 ----
    if dist.is_initialized():
        cnt = torch.tensor([tp, fp, tn, fn], device=device, dtype=torch.float32)
        dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
        tp, fp, tn, fn = cnt.tolist()

    total = tp + fp + tn + fn
    eps = 1e-8  # 防止除 0

    acc    = (tp + tn) / (total + eps)
    prec   = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1     = 2 * prec * recall / (prec + recall + eps)

    if rank == 0:
        print(f"[VAL]  Acc:{acc:.4f}  Prec:{prec:.4f}  "
              f"Recall:{recall:.4f}  F1:{f1:.4f}")
        print("TP:{}, FP:{}, TN:{}, FN:{}".format(tp, fp, tn, fn), '\n')
    return acc, prec, recall, f1

@torch.no_grad()
def validate_stroke(
        classifier,
        translator,
        loader,
        device,
        rank: int,
        num_classes: int = 3):
    """
    在 val_loader 上计算 Accuracy、Macro-Precision、Macro-Recall、Macro-F1
    （适用于多分类；这里默认 C=3）。

    Args
    ----
    classifier : nn.Module / DDP          # backbone+fc，输出 logits (B, C)
    loader     : DataLoader
    device     : 当前 rank 的 cuda
    rank       : 进程 rank (0 用来打印)
    num_classes: 类别数，默认 3
    """
    classifier.eval()

    # 每个类别的 TP / FP / FN
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)

    total = 0
    correct = 0

    for imgs, label in loader:               # label shape: (B,)
        imgs   = imgs.to(device)
        label  = label.to(device).long()     # {0,1,2}

        # 若你的模型只接受 3 通道，下面把单通道重复 3 次
        fake_mr = translator((imgs - 0.5) * 2)
        fake_mr = (fake_mr + 1) * 0.5
        logits = classifier(torch.cat([imgs, fake_mr, fake_mr], dim=1))

        # 若输出为 (B, 3, 1, 1) 之类 -> squeeze 掉额外维度
        if logits.ndim == 4:
            logits = logits.squeeze(-1).squeeze(-1)   # (B,3)

        pred = logits.argmax(dim=1)          # (B,)

        total   += label.numel()
        correct += (pred == label).sum().item()

        for c in range(num_classes):
            tp[c] += ((pred == c) & (label == c)).sum()
            fp[c] += ((pred == c) & (label != c)).sum()
            fn[c] += ((pred != c) & (label == c)).sum()

    # ---- DDP: 聚合计数 ----
    if dist.is_initialized():
        for tensor in (tp, fp, fn):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # 1. 封装成 tensor
        total_tensor = torch.tensor([total], device=device, dtype=torch.float32)
        correct_tensor = torch.tensor([correct], device=device, dtype=torch.float32)

        # 2. 就地求和
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)

        # 3. 取回标量
        total = total_tensor.item()
        correct = correct_tensor.item()

    eps = 1e-8
    acc = correct / (total + eps)

    prec_per_class   = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)
    f1_per_class     = 2 * prec_per_class * recall_per_class / (prec_per_class + recall_per_class + eps)

    # 宏平均（所有类别均等权）
    prec_macro   = prec_per_class.mean().item()
    recall_macro = recall_per_class.mean().item()
    f1_macro     = f1_per_class.mean().item()

    if rank == 0:
        print(f"[VAL]  Acc:{acc:.4f}  "
              f"Prec(macro):{prec_macro:.4f}  "
              f"Recall(macro):{recall_macro:.4f}  "
              f"F1(macro):{f1_macro:.4f}")
        for c in range(num_classes):
            print(f"  └─Class {c}: P={prec_per_class[c]:.4f} "
                  f"R={recall_per_class[c]:.4f} F1={f1_per_class[c]:.4f}")

    return acc, prec_macro, recall_macro, f1_macro


def main_worker(rank, world_size, args):
    torch.autograd.set_detect_anomaly(True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    best_f1 = 0.0
    best_acc = 0.0
    best_prec = 0.0
    best_recall = 0.0

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
        features_t.append(output)  # 将输出特征存储到列表中

    # dataset and loaders
    train_ds = ClassifyDatasetStroke(args.train_img)
    val_ds   = ClassifyDatasetStroke(args.val_img)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=1)
    val_loader   = DataLoader(val_ds, batch_size=1, sampler=val_sampler,  num_workers=1)

    # model
    # model_s = CenterNet(num_classes=1, n_channels=2).to(rank)
    # model_s = DDP(model_s, device_ids=[rank])
    # classifer = ResNet(input_channels=3).to(rank)
    classifer = ResNet18Backbone(in_channels=3, num_class=3).to(rank)
    classifer = DDP(classifer, device_ids=[rank])
    classifer_t = ResNet18Backbone(in_channels=3, num_class=3).to(rank)
    classifer_t = DDP(classifer_t, device_ids=[rank])
    # model_t = CenterNet(num_classes=1, n_channels=2).to(rank)
    # model_t = DDP(model_t, device_ids=[rank])
    translator = Generator().to(rank)
    translator.load_state_dict(torch.load('./Translator/logs/best_G.pth'))
    translator = DDP(translator, device_ids=[rank])


    register_hooks(classifer, hook_fn=hook_fn_s)
    register_hooks(classifer_t, hook_fn=hook_fn_t)

    optimizer_s = optim.Adam(classifer.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizer_t = optim.Adam(classifer_t.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizer_trans = optim.Adam(translator.parameters(), lr=LR, betas=(0.9, 0.999))

    # lr scheduler
    scheduler_s = optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=LR_STEP, gamma=GAMMA)
    scheduler_t = optim.lr_scheduler.MultiStepLR(optimizer_t, milestones=LR_STEP, gamma=GAMMA)
    scheduler_trans = optim.lr_scheduler.MultiStepLR(optimizer_trans, milestones=LR_STEP, gamma=GAMMA)

    for epoch in range(1, EPOCHS+1):
        train_sampler.set_epoch(epoch)

        classifer.train()
        classifer_t.train()
        translator.train()
        hm_loss = 0.0

        for i, (imgs, categroy) in enumerate(train_loader):
            imgs = imgs.to(rank)
            categroy = categroy.to(rank)

            optimizer_t.zero_grad()
            optimizer_trans.zero_grad()
            fake_mr_t = translator((imgs - 0.5) * 2)
            fake_mr_t = (fake_mr_t + 1) * 0.5
            prob_t = classifer_t(torch.cat([fake_mr_t, fake_mr_t, fake_mr_t], dim=1))
            # hm_pred_t = model_t(torch.cat([fake_mr, fake_mr], dim=1))
            # focal loss (BCE) for heatmap
            # loss_hm_t = F.binary_cross_entropy(hm_pred_t, hm_gt) + F.mse_loss(hm_pred_t, hm_gt)
            # loss_hm_t.backward()
            loss_class_t = F.cross_entropy(prob_t, categroy.squeeze(-1))
            loss_class_t.backward()


            optimizer_s.zero_grad()
            optimizer_trans.zero_grad()
            # hm_pred = model_s(torch.cat([imgs, imgs], dim=1))
            # focal loss (BCE) for heatmap
            # loss_hm = F.binary_cross_entropy(hm_pred, hm_gt) + F.mse_loss(hm_pred, hm_gt)
            fake_mr = translator((imgs - 0.5) * 2)
            fake_mr = (fake_mr + 1) * 0.5
            prob = classifer(torch.cat([imgs, fake_mr, fake_mr], dim=1))
            loss_class = F.cross_entropy(prob, categroy.squeeze(-1))

            if loss_class > loss_class_t:
                loss_distill = Distillation_loss(features_s, features_t)
            else:
                loss_distill = 0.0


            # 取第0张图的 GT & Pred
            # gt = hm_gt[0, 0].cpu().detach().numpy()  # [fh,fw]
            # pr = hm_pred[0, 0].cpu().detach().numpy()  # [fh,fw]
            # # 并排可视化
            # plt.figure(figsize=(8, 4))
            # plt.subplot(1, 2, 1)
            # plt.title("GT Heatmap")
            # plt.imshow(gt, cmap='jet')
            # plt.colorbar(fraction=0.046, pad=0.04)
            # plt.axis('off')
            # plt.subplot(1, 2, 2)
            # plt.title("Pred Heatmap")
            # plt.imshow(pr, cmap='jet')
            # plt.colorbar(fraction=0.046, pad=0.04)
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()
            # if i == 0:
            #     plt.close()


            loss =  loss_class + loss_distill
            loss.backward()

            optimizer_trans.step()
            optimizer_t.step()
            optimizer_s.step()

            features_t.clear()
            features_s.clear()

            hm_loss += loss_class.item()

        if rank == 0:
            print(f"Epoch {epoch} class loss: {hm_loss / len(train_loader):.4f} ")

        with torch.no_grad():
            acc, prec, recall, f1 = validate_stroke(classifer, translator, val_loader, rank=rank, device=rank)
            if f1 >= best_f1:
                best_f1 = f1
                if rank == 0:
                    print('best_f1:{}, epoch{}'.format(best_f1, epoch))
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    torch.save(classifer.module.state_dict(), os.path.join(args.save_dir, 'best_cls.pth'))
                    torch.save(translator.module.state_dict(), os.path.join(args.save_dir, 'best_trans.pth'))

        features_t.clear()
        features_s.clear()

        scheduler_s.step()
        scheduler_t.step()
        scheduler_trans.step()

        gc.collect()
        torch.cuda.empty_cache()


        # save best
        # if rank == 0:
            # torch.save(model_s.module.state_dict(), os.path.join(args.save_dir, '{}_s.pth'.format(epoch)))
            # torch.save(classifer.module.state_dict(), os.path.join(args.save_dir, '{}_cls_s.pth'.format(epoch)))

            # torch.save(model_t.module.state_dict(), os.path.join(args.save_dir, '{}_t.pth'.format(epoch)))
            # torch.save(translator.module.state_dict(), os.path.join(args.save_dir, '{}_g.pth'.format(epoch)))

        torch.cuda.ipc_collect()
    dist.destroy_process_group()



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_img', default='./data4cls/train/stroke_img')
    p.add_argument('--train_ann', default='./data4cls/train/annotations')
    p.add_argument('--val_img',   default='./data4cls/val/stroke_img')
    p.add_argument('--val_ann',   default='./data4cls/val/annotations')
    p.add_argument('--save_dir',  default='./logs/cls_stroke_distill')
    p.add_argument('--world_size', type=int, default=torch.cuda.device_count())
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    world_size = args.world_size
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
