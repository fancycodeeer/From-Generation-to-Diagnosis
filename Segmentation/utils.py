import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        if 'LOCAL_RANK' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
        elif hasattr(args, 'local_rank') and args.local_rank >= 0:
            args.local_rank = int(args.local_rank)
        else:
            args.local_rank = 0
    elif hasattr(args, 'local_rank') and args.local_rank >= 0:
        args.distributed = True
        args.rank = int(os.environ.get('RANK', 0))
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        args.local_rank = int(args.local_rank)
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    if args.distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device('cuda', args.local_rank)
            backend = 'nccl'
        else:
            args.device = torch.device('cpu')
            backend = 'gloo'
        dist.init_process_group(backend=backend, init_method='env://', rank=args.rank, world_size=args.world_size)
        dist.barrier()
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def is_main_process(args=None):
    if args is not None:
        return getattr(args, 'rank', 0) == 0
    if not is_dist_avail_and_initialized():
        return True
    return dist.get_rank() == 0

def ddp_barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def set_seed(seed: int, rank: int=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_config(args, save_path: str):
    ensure_dir(str(Path(save_path).parent))
    config = vars(args).copy()
    for k, v in list(config.items()):
        if isinstance(v, torch.device):
            config[k] = str(v)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def remove_module_prefix(state_dict: Dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        new_state[k] = v
    return new_state

def get_raw_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model

def save_checkpoint(model, optimizer, epoch: int, best_dice: float, best_iou: float, save_path: str):
    ensure_dir(str(Path(save_path).parent))
    raw_model = get_raw_model(model)
    ckpt = {'epoch': epoch, 'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_dice': best_dice, 'best_iou': best_iou}
    torch.save(ckpt, save_path)

def load_checkpoint(model, optimizer, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = remove_module_prefix(ckpt['model'])
    raw_model = get_raw_model(model)
    raw_model.load_state_dict(state_dict, strict=True)
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = int(ckpt['epoch']) + 1
    best_dice = float(ckpt.get('best_dice', -1.0))
    best_iou = float(ckpt.get('best_iou', -1.0))
    return (start_epoch, best_dice, best_iou)

def reduce_train_losses(loss_sums: Dict[str, float], num_batches: int, device):
    keys = ['loss_total', 'loss_bce', 'loss_dice', 'loss_iou', 'loss_focal']
    values = [loss_sums[k] for k in keys]
    values.append(float(num_batches))
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    global_num_batches = max(float(tensor[-1].item()), 1.0)
    avg_losses = {}
    for i, k in enumerate(keys):
        avg_losses[k] = float(tensor[i].item() / global_num_batches)
    return avg_losses

def safe_divide(numerator: float, denominator: float, empty_value: float=1.0):
    if denominator == 0:
        return empty_value
    return numerator / denominator

def save_case_metrics_csv(case_metrics: List[Dict], save_path: str):
    ensure_dir(str(Path(save_path).parent))
    fieldnames = ['case_id', 'dice', 'iou', 'precision', 'recall', 'tp', 'fp', 'fn']
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_metrics:
            writer.writerow(row)

def append_epoch_metrics_csv(row: Dict, save_path: str):
    ensure_dir(str(Path(save_path).parent))
    file_exists = Path(save_path).exists()
    fieldnames = ['epoch', 'lr', 'train_loss', 'train_bce', 'train_soft_dice_loss', 'train_soft_iou_loss', 'train_focal', 'case_dice', 'case_iou', 'case_precision', 'case_recall', 'num_cases', 'is_best']
    with open(save_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def is_best_epoch(current_dice: float, current_iou: float, best_dice: float, best_iou: float):
    if current_dice > best_dice:
        return True
    if abs(current_dice - best_dice) < 1e-12 and current_iou > best_iou:
        return True
    return False

def print_epoch_log(epoch, total_epochs, lr_now, train_losses, eval_summary, best_dice, best_iou, is_best):
    print(f"[Epoch {epoch:03d}/{total_epochs}] lr={lr_now:.2e} | training loss={train_losses['loss_total']:.6f} BCE={train_losses['loss_bce']:.6f} DiceLoss={train_losses['loss_dice']:.6f} IoULoss={train_losses['loss_iou']:.6f} Focal={train_losses['loss_focal']:.6f} | case Dice={eval_summary['dice']:.6f} IoU={eval_summary['iou']:.6f} Precision={eval_summary['precision']:.6f} Recall={eval_summary['recall']:.6f} | best Dice={best_dice:.6f} best IoU={best_iou:.6f} {(' <-- best' if is_best else '')}")

def print_train_header(args, device):
    print('=' * 80)
    print('BraTS2021 binary segmentation training')
    print(f'DDP: {args.distributed}')
    print(f'rank/world_size: {args.rank}/{args.world_size}')
    print(f'local_rank: {args.local_rank}')
    print(f'Device: {device}')
    print(f'Data root: {args.dataroot}')
    print(f'Training split: {args.train_phase}')
    print(f'Evaluation split: {args.eval_phase}')
    print(f'Single-modality input: {args.modality}')
    print('MRI input range: [-1, 1]')
    print('Mask range: {0, 1}')
    print('loss: BCE + soft Dice + soft IoU + FocalLoss')
    print('=' * 80)

def print_train_finish(args, best_dice, best_iou):
    print('=' * 80)
    print('Training complete')
    print(f'best case-level Dice: {best_dice:.6f}')
    print(f'best case-level IoU : {best_iou:.6f}')
    print(f"Metrics file: {os.path.join(args.save_dir, 'metrics_epoch.csv')}")
    print(f"Best model: {os.path.join(args.save_dir, 'checkpoints', 'best.pth')}")
    print('=' * 80)
