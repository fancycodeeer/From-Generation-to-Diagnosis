import argparse
import os
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast
from dataset import BraTS2021BinarySegDataset
from loss import ComboSegLoss
from models import UNetCoordinateChannelAttention
from utils import append_epoch_metrics_csv, cleanup_distributed, ddp_barrier, ensure_dir, get_raw_model, init_distributed_mode, is_main_process, load_checkpoint, print_epoch_log, print_train_finish, print_train_header, save_case_metrics_csv, save_checkpoint, save_config, set_seed, defaultdict, safe_divide, reduce_train_losses

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--train_phase', type=str, default='train', help='Training split name')
    parser.add_argument('--eval_phase', type=str, default='val')
    parser.add_argument('--modality', type=str, default='t1', choices=['t1', 't1ce'], help='Single input modality')
    parser.add_argument('--load_size', type=int, default=0, help='Resize size; 0 disables resizing')
    parser.add_argument('--crop_size', type=int, default=0, help='Random crop size; 0 disables cropping')
    parser.add_argument('--no_flip', action='store_true', help='Disable horizontal flipping during training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker count per DDP process')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=2026, help='Random seed')
    parser.add_argument('--amp', action='store_true', help='Enable mixed-precision training')
    parser.add_argument('--threshold', type=float, default=0.5, help='Evaluation binarization threshold')
    parser.add_argument('--patience', type=int, default=20, help='Early-stopping patience based on case-level Dice')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum Dice improvement for early stopping')
    parser.add_argument('--base_channels', type=int, default=64, help='U-Net base channel count')
    parser.add_argument('--find_unused_parameters', action='store_true', help='Enable DDP unused-parameter discovery')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1, help='Local GPU index supplied by torchrun')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--resume', type=str, default='', help='Resume checkpoint path; empty starts from scratch')
    parser.add_argument('--save_every', type=int, default=3, help='Periodic checkpoint interval; nonpositive disables periodic checkpoints')
    return parser.parse_args()

def build_model(args):
    model = UNetCoordinateChannelAttention(in_channels=2, num_classes=1, base_channels=args.base_channels, bilinear=True, use_attention=True)
    return model

def build_dataloader(args):
    train_dataset = BraTS2021BinarySegDataset(dataroot=args.dataroot, phase=args.train_phase, modality=args.modality, load_size=args.load_size, crop_size=args.crop_size, no_flip=args.no_flip)
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=False)
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False, persistent_workers=args.num_workers > 0)
    if is_main_process(args):
        eval_dataset = BraTS2021BinarySegDataset(dataroot=args.dataroot, phase=args.eval_phase, modality=args.modality, load_size=args.load_size, crop_size=0, no_flip=True)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False, persistent_workers=args.num_workers > 0)
    else:
        eval_loader = None
    return (train_loader, eval_loader, train_sampler)

@torch.no_grad()
def evaluate_case_level(model, loader, device, threshold: float=0.5):
    model.eval()
    case_stats = defaultdict(lambda: {'tp': 0.0, 'fp': 0.0, 'fn': 0.0})
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        case_ids = batch['case_id']
        logits = model(torch.cat([images, images], dim=1))
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        preds = preds.detach().cpu()
        masks = masks.detach().cpu()
        batch_size = preds.shape[0]
        for i in range(batch_size):
            pred_i = preds[i]
            mask_i = masks[i]
            tp = torch.sum((pred_i == 1) & (mask_i == 1)).item()
            fp = torch.sum((pred_i == 1) & (mask_i == 0)).item()
            fn = torch.sum((pred_i == 0) & (mask_i == 1)).item()
            case_id = str(case_ids[i])
            case_stats[case_id]['tp'] += tp
            case_stats[case_id]['fp'] += fp
            case_stats[case_id]['fn'] += fn
    case_metrics = []
    for case_id, s in sorted(case_stats.items()):
        tp = s['tp']
        fp = s['fp']
        fn = s['fn']
        dice = safe_divide(2.0 * tp, 2.0 * tp + fp + fn, empty_value=1.0)
        iou = safe_divide(tp, tp + fp + fn, empty_value=1.0)
        precision = safe_divide(tp, tp + fp, empty_value=1.0)
        recall = safe_divide(tp, tp + fn, empty_value=1.0)
        case_metrics.append({'case_id': case_id, 'dice': dice, 'iou': iou, 'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn})
    if len(case_metrics) == 0:
        raise RuntimeError('No cases were available during evaluation.')
    summary = {'dice': float(np.mean([m['dice'] for m in case_metrics])), 'iou': float(np.mean([m['iou'] for m in case_metrics])), 'precision': float(np.mean([m['precision'] for m in case_metrics])), 'recall': float(np.mean([m['recall'] for m in case_metrics])), 'num_cases': len(case_metrics)}
    return (summary, case_metrics)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp: bool=False):
    model.train()
    loss_sums = {'loss_total': 0.0, 'loss_bce': 0.0, 'loss_dice': 0.0, 'loss_iou': 0.0, 'loss_focal': 0.0}
    num_batches = 0
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
                logits = model(torch.cat([images, images], dim=1))
                loss, loss_items = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(torch.cat([images, images], dim=1))
            loss, loss_items = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        for k in loss_sums:
            loss_sums[k] += loss_items[k]
        num_batches += 1
    avg_losses = reduce_train_losses(loss_sums=loss_sums, num_batches=num_batches, device=device)
    return avg_losses

def main():
    args = get_args()
    missing = [name for name, value in {'dataroot': args.dataroot, 'save_dir': args.save_dir}.items() if value == '']
    if missing:
        raise ValueError('Set required arguments: ' + ', '.join(missing))
    args = init_distributed_mode(args)
    set_seed(args.seed, rank=args.rank)
    if is_main_process(args):
        ensure_dir(args.save_dir)
        ensure_dir(os.path.join(args.save_dir, 'case_metrics'))
        ensure_dir(os.path.join(args.save_dir, 'checkpoints'))
        save_config(args, os.path.join(args.save_dir, 'config.json'))
    ddp_barrier()
    device = args.device
    if is_main_process(args):
        print_train_header(args, device)
        print(f'Early stopping: patience={args.patience} | min_delta={args.min_delta}')
    train_loader, eval_loader, train_sampler = build_dataloader(args)
    model = build_model(args).to(device)
    if args.distributed:
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=args.find_unused_parameters, broadcast_buffers=True)
        else:
            model = DDP(model, find_unused_parameters=args.find_unused_parameters, broadcast_buffers=True)
    criterion = ComboSegLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scaler = GradScaler(enabled=args.amp)
    start_epoch = 1
    best_dice = -1.0
    best_iou = -1.0
    best_epoch = 0
    patience_counter = 0
    if args.resume:
        if is_main_process(args):
            print(f'Loading checkpoint: {args.resume}')
        start_epoch, best_dice, best_iou = load_checkpoint(model=model, optimizer=optimizer, ckpt_path=args.resume, device=device)
        best_epoch = start_epoch - 1
        patience_counter = 0
        if is_main_process(args):
            print(f'Resuming from epoch {start_epoch} for continued training.')
            print(f'Current best Dice={best_dice:.6f}, best IoU={best_iou:.6f}')
            print('The patience counter is reset after resume.')
    ddp_barrier()
    metrics_csv = os.path.join(args.save_dir, 'metrics_epoch.csv')
    for epoch in range(start_epoch, args.epochs + 1):
        should_stop = False
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        lr_now = optimizer.param_groups[0]['lr']
        train_losses = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, device=device, scaler=scaler, use_amp=args.amp)
        if is_main_process(args):
            eval_model = get_raw_model(model)
            eval_summary, case_metrics = evaluate_case_level(model=eval_model, loader=eval_loader, device=device, threshold=args.threshold)
            case_csv_path = os.path.join(args.save_dir, 'case_metrics', f'case_metrics_epoch_{epoch:03d}.csv')
            save_case_metrics_csv(case_metrics, case_csv_path)
            current_dice = eval_summary['dice']
            current_iou = eval_summary['iou']
            is_best = current_dice > best_dice + args.min_delta
            if is_best:
                best_dice = current_dice
                best_iou = current_iou
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, best_dice=best_dice, best_iou=best_iou, save_path=os.path.join(args.save_dir, 'checkpoints', 'best.pth'))
            else:
                patience_counter += 1
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, best_dice=best_dice, best_iou=best_iou, save_path=os.path.join(args.save_dir, 'checkpoints', 'latest.pth'))
            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, best_dice=best_dice, best_iou=best_iou, save_path=os.path.join(args.save_dir, 'checkpoints', f'epoch_{epoch:03d}.pth'))
            append_epoch_metrics_csv({'epoch': epoch, 'lr': lr_now, 'train_loss': train_losses['loss_total'], 'train_bce': train_losses['loss_bce'], 'train_soft_dice_loss': train_losses['loss_dice'], 'train_soft_iou_loss': train_losses['loss_iou'], 'train_focal': train_losses['loss_focal'], 'case_dice': eval_summary['dice'], 'case_iou': eval_summary['iou'], 'case_precision': eval_summary['precision'], 'case_recall': eval_summary['recall'], 'num_cases': eval_summary['num_cases'], 'is_best': int(is_best)}, metrics_csv)
            print_epoch_log(epoch=epoch, total_epochs=args.epochs, lr_now=lr_now, train_losses=train_losses, eval_summary=eval_summary, best_dice=best_dice, best_iou=best_iou, is_best=is_best)
            print(f'Early-stopping status: patience={patience_counter}/{args.patience} | best_epoch={best_epoch} | best Dice={best_dice:.6f} | Current Dice={current_dice:.6f}')
            should_stop = patience_counter >= args.patience
            if should_stop:
                print(f'Early stopping triggered after {args.patience} epochs without improvement.')
        if args.distributed:
            if is_main_process(args):
                stop_tensor = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int32)
            else:
                stop_tensor = torch.tensor([0], device=device, dtype=torch.int32)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        ddp_barrier()
        if should_stop:
            break
    if is_main_process(args):
        print_train_finish(args, best_dice, best_iou)
    cleanup_distributed()
if __name__ == '__main__':
    main()
