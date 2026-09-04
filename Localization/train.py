import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from datasets import DeepLesionHeatmapDataset, heatmap_collate
from losses import HeatmapLoss
from models import UNetSECoord

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def setup_ddp(args) -> Tuple[bool, int, int, int]:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
        elif getattr(args, 'local_rank', -1) >= 0:
            local_rank = int(args.local_rank)
        else:
            local_rank = rank % torch.cuda.device_count()
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return (True, rank, local_rank, world_size)
    return (False, 0, 0, 1)

def cleanup_ddp(is_ddp: bool):
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()

def is_main(rank: int) -> bool:
    return rank == 0

def all_reduce_sum(x: float, device: torch.device, is_ddp: bool) -> float:
    t = torch.tensor([float(x)], dtype=torch.float64, device=device)
    if is_ddp:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())

def reduce_numeric_mean_from_df(df: pd.DataFrame, col: str, device: torch.device, is_ddp: bool) -> float:
    if df is None or len(df) == 0 or col not in df.columns:
        local_sum = 0.0
        local_count = 0.0
    else:
        vals = pd.to_numeric(df[col], errors='coerce')
        vals = vals.replace([np.inf, -np.inf], np.nan)
        valid = vals.notna()
        local_sum = float(vals[valid].sum()) if valid.any() else 0.0
        local_count = float(valid.sum())
    global_sum = all_reduce_sum(local_sum, device, is_ddp)
    global_count = all_reduce_sum(local_count, device, is_ddp)
    if global_count <= 0:
        return float('nan')
    return float(global_sum / global_count)

def gather_rows(rows: List[Dict[str, Any]], is_ddp: bool) -> List[Dict[str, Any]]:
    return rows

def broadcast_object(obj: Any, is_ddp: bool, src: int=0) -> Any:
    if not is_ddp:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]

class DistributedEvalSampler(Sampler):

    def __init__(self, dataset, num_replicas: Optional[int]=None, rank: Optional[int]=None):
        self.dataset = dataset
        self.num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
        self.rank = dist.get_rank() if rank is None else rank
        self.indices = list(range(self.rank, len(self.dataset), self.num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

def make_train_loader(dataset, batch_size: int, num_workers: int, is_ddp: bool, rank: int, world_size: int):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if is_ddp else None
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None, num_workers=num_workers, pin_memory=True, collate_fn=heatmap_collate, drop_last=False)
    return (loader, sampler)

def make_eval_loader(dataset, batch_size: int, num_workers: int, is_ddp: bool, rank: int, world_size: int):
    sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank) if is_ddp else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=heatmap_collate, drop_last=False)

def extract_peaks_2d(arr: np.ndarray, threshold: float, nms_kernel: int, max_peaks: int) -> List[Dict[str, float]]:
    if arr.ndim != 2:
        raise ValueError(f'extract_peaks_2d expects 2D array, got {arr.shape}')
    nms_kernel = max(1, int(nms_kernel))
    if nms_kernel % 2 == 0:
        nms_kernel += 1
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(t, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2).squeeze().numpy()
    mask = (arr >= float(threshold)) & (arr == pooled)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    scores = arr[ys, xs]
    order = np.argsort(-scores)
    min_dist = max(1.0, nms_kernel / 2.0)
    peaks: List[Dict[str, float]] = []
    for j in order:
        x = float(xs[j])
        y = float(ys[j])
        score = float(scores[j])
        keep = True
        for p in peaks:
            if ((x - p['x']) ** 2 + (y - p['y']) ** 2) ** 0.5 < min_dist:
                keep = False
                break
        if keep:
            peaks.append({'x': x, 'y': y, 'score': score})
        if max_peaks > 0 and len(peaks) >= max_peaks:
            break
    return peaks

def match_peaks(pred_peaks: List[Dict[str, float]], gt_peaks: List[Dict[str, float]], radius: float) -> Tuple[int, int, int, float]:
    if len(gt_peaks) == 0:
        return (0, len(pred_peaks), 0, float('nan'))
    if len(pred_peaks) == 0:
        return (0, 0, len(gt_peaks), float('nan'))
    candidates = []
    for pi, p in enumerate(pred_peaks):
        for gi, g in enumerate(gt_peaks):
            d = ((p['x'] - g['x']) ** 2 + (p['y'] - g['y']) ** 2) ** 0.5
            if d <= radius:
                candidates.append((d, pi, gi))
    candidates.sort(key=lambda x: x[0])
    used_p, used_g, dists = (set(), set(), [])
    for d, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        dists.append(float(d))
    tp = len(dists)
    fp = len(pred_peaks) - tp
    fn = len(gt_peaks) - tp
    mean_dist = float(np.mean(dists)) if len(dists) else float('nan')
    return (tp, fp, fn, mean_dist)

def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0

def fp_target_to_name(x: float) -> str:
    s = f'{float(x):g}'
    return s.replace('.', '_').replace('-', 'm')

def peaks_to_json(peaks: List[Dict[str, float]]) -> str:
    compact = []
    for p in peaks:
        compact.append([float(p['x']), float(p['y']), float(p.get('score', 1.0))])
    return json.dumps(compact, separators=(',', ':'))

def peaks_from_json(s: Any) -> List[Dict[str, float]]:
    if s is None:
        return []
    if isinstance(s, float) and np.isnan(s):
        return []
    s = str(s)
    if len(s) == 0:
        return []
    try:
        raw = json.loads(s)
    except Exception:
        return []
    peaks: List[Dict[str, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        score = float(item[2]) if len(item) >= 3 else 1.0
        peaks.append({'x': float(item[0]), 'y': float(item[1]), 'score': score})
    return peaks

def compute_froc_metrics_for_group(g: pd.DataFrame, args) -> Dict[str, float]:
    targets = [float(x) for x in getattr(args, 'froc_fp_targets', [0.5, 1.0, 2.0])]
    out = {f'sens_at_{fp_target_to_name(t)}fp': float('nan') for t in targets}
    if g is None or len(g) == 0:
        return out
    image_records: List[Dict[str, Any]] = []
    candidates: List[Dict[str, float]] = []
    total_gt = 0
    for image_i, (_, row) in enumerate(g.iterrows()):
        gt_peaks = peaks_from_json(row.get('_froc_gt_peaks_json', '[]'))
        pred_peaks = peaks_from_json(row.get('_froc_pred_peaks_json', '[]'))
        image_records.append({'gt_peaks': gt_peaks, 'used_gt': set()})
        total_gt += len(gt_peaks)
        for p in pred_peaks:
            candidates.append({'image_i': int(image_i), 'x': float(p['x']), 'y': float(p['y']), 'score': float(p.get('score', 1.0))})
    if total_gt <= 0:
        return out
    num_images = max(int(len(g)), 1)
    candidates.sort(key=lambda p: (-float(p['score']), float(p['y']), float(p['x'])))
    max_fp_allowed = {t: float(t) * float(num_images) for t in targets}
    best_tp = {t: 0 for t in targets}
    tp = 0
    fp = 0
    for p in candidates:
        rec = image_records[int(p['image_i'])]
        gt_peaks = rec['gt_peaks']
        used_gt = rec['used_gt']
        best_gi = None
        best_dist = float('inf')
        for gi, q in enumerate(gt_peaks):
            if gi in used_gt:
                continue
            d = ((float(p['x']) - float(q['x'])) ** 2 + (float(p['y']) - float(q['y'])) ** 2) ** 0.5
            if d <= float(args.match_radius) and d < best_dist:
                best_dist = d
                best_gi = gi
        if best_gi is None:
            fp += 1
        else:
            used_gt.add(best_gi)
            tp += 1
        for t in targets:
            if float(fp) <= max_fp_allowed[t] + 1e-12:
                best_tp[t] = max(best_tp[t], tp)
    for t in targets:
        out[f'sens_at_{fp_target_to_name(t)}fp'] = safe_div(best_tp[t], total_gt)
    return out

def get_froc_metric_cols(args) -> List[str]:
    return [f'sens_at_{fp_target_to_name(float(t))}fp' for t in getattr(args, 'froc_fp_targets', [0.5, 1.0, 2.0])]

@torch.no_grad()
def compute_batch_metrics(prob: torch.Tensor, gt: torch.Tensor, metas: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    eps = 1e-08
    b, _, h, w = prob.shape
    p = prob[:, 0].reshape(b, -1).clamp(eps, 1.0 - eps)
    g = gt[:, 0].reshape(b, -1).clamp(0.0, 1.0)
    mse = ((p - g) ** 2).mean(dim=1)
    mae = (p - g).abs().mean(dim=1)
    bce = F.binary_cross_entropy(p, g, reduction='none').mean(dim=1)
    p_dist = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    g_dist = g / g.sum(dim=1, keepdim=True).clamp_min(eps)
    kl = (g_dist * (torch.log(g_dist + eps) - torch.log(p_dist + eps))).sum(dim=1)
    pred_idx = p.argmax(dim=1)
    gt_idx = g.argmax(dim=1)
    pred_y = torch.div(pred_idx, w, rounding_mode='floor').float()
    pred_x = (pred_idx % w).float()
    gt_y = torch.div(gt_idx, w, rounding_mode='floor').float()
    gt_x = (gt_idx % w).float()
    prob_np = prob[:, 0].detach().float().cpu().numpy()
    gt_np = gt[:, 0].detach().float().cpu().numpy()
    miss_distance = float((h ** 2 + w ** 2) ** 0.5)
    rows: List[Dict[str, Any]] = []
    for i in range(b):
        pred_peak_threshold = float(args.pred_peak_threshold) * float(np.max(prob_np[i]))
        pred_peaks = extract_peaks_2d(prob_np[i], pred_peak_threshold, args.peak_nms_kernel, args.max_pred_peaks)
        gt_peaks = extract_peaks_2d(gt_np[i], args.gt_peak_threshold, args.peak_nms_kernel, args.max_gt_peaks)
        froc_pred_peaks = extract_peaks_2d(prob_np[i], float(args.froc_min_peak_score), args.peak_nms_kernel, args.max_froc_pred_peaks)
        tp, fp, fn, mean_match_dist = match_peaks(pred_peaks, gt_peaks, args.match_radius)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        if len(pred_peaks) > 0 and len(gt_peaks) > 0:
            top_pred = pred_peaks[0]
            dists = [((top_pred['x'] - q['x']) ** 2 + (top_pred['y'] - q['y']) ** 2) ** 0.5 for q in gt_peaks]
            top_peak_error = float(min(dists))
        elif len(gt_peaks) > 0:
            top_peak_error = miss_distance
        else:
            top_peak_error = 0.0 if len(pred_peaks) == 0 else miss_distance
        rows.append({'mse': float(mse[i].item()), 'mae': float(mae[i].item()), 'bce': float(bce[i].item()), 'kl': float(kl[i].item()), 'top_peak_error_px': float(top_peak_error), 'top_hit5': float(top_peak_error <= 5.0), 'top_hit10': float(top_peak_error <= 10.0), 'top_hit20': float(top_peak_error <= 20.0), 'num_gt_peaks': int(len(gt_peaks)), 'num_pred_peaks': int(len(pred_peaks)), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'mean_match_dist_px': float(mean_match_dist) if np.isfinite(mean_match_dist) else float('nan'), 'pred_peak_x': float(pred_x[i].item()), 'pred_peak_y': float(pred_y[i].item()), 'gt_peak_x': float(gt_x[i].item()), 'gt_peak_y': float(gt_y[i].item()), 'pred_peak_value': float(p[i, pred_idx[i]].item()), 'gt_peak_value': float(g[i, gt_idx[i]].item()), '_froc_pred_peaks_json': peaks_to_json(froc_pred_peaks), '_froc_gt_peaks_json': peaks_to_json(gt_peaks)})
    return rows

def summarize_case_metrics(slice_df: pd.DataFrame, args=None) -> pd.DataFrame:
    if len(slice_df) == 0:
        return pd.DataFrame()
    mean_cols = ['mse', 'mae', 'bce', 'kl', 'top_peak_error_px', 'top_hit5', 'top_hit10', 'top_hit20', 'mean_match_dist_px', 'pred_peak_value', 'gt_peak_value', 'num_gt_peaks', 'num_pred_peaks']
    rows = []
    for case_id, g in slice_df.groupby('case_id', sort=False):
        row: Dict[str, Any] = {'case_id': str(case_id), 'patient_id': str(g['patient_id'].iloc[0]) if 'patient_id' in g.columns else str(case_id), 'n_slices': int(len(g)), 'tp': int(g['tp'].sum()), 'fp': int(g['fp'].sum()), 'fn': int(g['fn'].sum())}
        row['precision'] = safe_div(row['tp'], row['tp'] + row['fp'])
        row['recall'] = safe_div(row['tp'], row['tp'] + row['fn'])
        row['f1'] = safe_div(2.0 * row['precision'] * row['recall'], row['precision'] + row['recall'])
        for col in mean_cols:
            if col in g.columns:
                row[col] = float(pd.to_numeric(g[col], errors='coerce').mean())
        if args is not None and '_froc_pred_peaks_json' in g.columns and ('_froc_gt_peaks_json' in g.columns):
            row.update(compute_froc_metrics_for_group(g, args))
        rows.append(row)
    return pd.DataFrame(rows)

def normalize_monitor_name(metric_name: str) -> str:
    metric_name = str(metric_name)
    if metric_name.startswith('val_'):
        metric_name = metric_name[4:]
    if metric_name.startswith('test_'):
        metric_name = metric_name[5:]
    return metric_name

def metric_higher_is_better(metric_name: str) -> bool:
    metric_name = normalize_monitor_name(metric_name)
    return any((x in metric_name for x in ['f1', 'precision', 'recall', 'sens', 'hit']))

def get_initial_best(monitor: str) -> float:
    return -float('inf') if metric_higher_is_better(monitor) else float('inf')

def is_improved(current: float, best: float, monitor: str) -> bool:
    if not np.isfinite(current):
        return False
    if metric_higher_is_better(monitor):
        return current > best
    return current < best

def sanitize_metric_name(metric_name: str) -> str:
    metric_name = normalize_monitor_name(metric_name)
    return re.sub('[^A-Za-z0-9_.-]+', '_', metric_name)

def fmt_float(x: Any, digits: int=5) -> str:
    try:
        x = float(x)
    except Exception:
        return 'nan'
    if not np.isfinite(x):
        return 'nan'
    return f'{x:.{digits}f}'

def save_model_only(path: Path, model):
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(raw_model.state_dict(), path)

def load_model_only(path: Path, model, device: torch.device):
    raw_model = model.module if isinstance(model, DDP) else model
    state = torch.load(path, map_location=device)
    raw_model.load_state_dict(state, strict=True)

def train_one_epoch(model, loader, sampler, optimizer, scaler, criterion, device, epoch: int, args, is_ddp: bool):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    amp_enabled = not args.no_amp and device.type == 'cuda'
    total_loss, total_focal, total_mse, total_n = (0.0, 0.0, 0.0, 0)
    for imgs, hms, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        hms = hms.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            logits = model(imgs)
            loss, comps = criterion(logits, hms)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        total_focal += float(comps['focal'].item()) * bs
        total_mse += float(comps['weighted_mse'].item()) * bs
        total_n += bs
    total_loss = all_reduce_sum(total_loss, device, is_ddp)
    total_focal = all_reduce_sum(total_focal, device, is_ddp)
    total_mse = all_reduce_sum(total_mse, device, is_ddp)
    total_n = all_reduce_sum(total_n, device, is_ddp)
    return {'step_loss': total_loss / max(total_n, 1.0), 'step_focal': total_focal / max(total_n, 1.0), 'step_weighted_mse': total_mse / max(total_n, 1.0)}

@torch.no_grad()
def evaluate_split(model, loader, criterion, device, epoch: int, split: str, out_dir: str, args, rank: int, is_ddp: bool):
    model.eval()
    amp_enabled = not args.no_amp and device.type == 'cuda'
    total_loss, total_n, rows = (0.0, 0, [])
    for imgs, hms, metas in loader:
        imgs = imgs.to(device, non_blocking=True)
        hms = hms.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(imgs)
            loss, _ = criterion(logits, hms)
            prob = torch.sigmoid(logits)
        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        metric_rows = compute_batch_metrics(prob.float(), hms.float(), metas, args)
        for m, meta in zip(metric_rows, metas):
            row = {'epoch': int(epoch), 'split': split, 'case_id': str(meta['case_id']), 'patient_id': str(meta['patient_id']), 'series_id': str(meta.get('series_id', '')), 'slice_id': str(meta.get('slice_id', '')), 'file_name': str(meta['file_name']), 'image_path': str(meta['image_path']), 'heatmap_path': str(meta['heatmap_path'])}
            row.update(m)
            rows.append(row)
    total_loss = all_reduce_sum(total_loss, device, is_ddp)
    total_n = all_reduce_sum(total_n, device, is_ddp)
    metrics_dir = Path(out_dir) / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        world_size = dist.get_world_size()
        tmp_dir = metrics_dir / '_rank_tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_slice_path = tmp_dir / f'epoch_{epoch:03d}_{split}_slice_rank{rank:03d}.csv'
        pd.DataFrame(rows).to_csv(tmp_slice_path, index=False, encoding='utf-8-sig')
        dist.barrier()
        summary = None
        if is_main(rank):
            slice_parts = []
            for r in range(world_size):
                p = tmp_dir / f'epoch_{epoch:03d}_{split}_slice_rank{r:03d}.csv'
                if not p.exists():
                    raise RuntimeError(f'Missing rank CSV: {p}')
                try:
                    part = pd.read_csv(p)
                except pd.errors.EmptyDataError:
                    part = pd.DataFrame()
                if len(part) > 0:
                    slice_parts.append(part)
            if len(slice_parts) > 0:
                slice_df = pd.concat(slice_parts, ignore_index=True)
            else:
                slice_df = pd.DataFrame()
            case_df = summarize_case_metrics(slice_df, args)
            slice_df.to_csv(metrics_dir / f'epoch_{epoch:03d}_{split}_slice.csv', index=False, encoding='utf-8-sig')
            if len(case_df) > 0:
                case_df.insert(1, 'epoch', int(epoch))
                case_df.insert(2, 'split', split)
            case_df.to_csv(metrics_dir / f'epoch_{epoch:03d}_{split}_case.csv', index=False, encoding='utf-8-sig')
            for r in range(world_size):
                p = tmp_dir / f'epoch_{epoch:03d}_{split}_slice_rank{r:03d}.csv'
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass
            summary = {'loss': total_loss / max(total_n, 1.0)}
            metric_cols = ['mse', 'mae', 'bce', 'kl', 'top_peak_error_px', 'top_hit5', 'top_hit10', 'top_hit20', 'num_gt_peaks', 'num_pred_peaks', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_match_dist_px', 'pred_peak_value', 'gt_peak_value'] + get_froc_metric_cols(args)
            for col in metric_cols:
                if col in slice_df.columns:
                    summary[f'slice_{col}'] = float(pd.to_numeric(slice_df[col], errors='coerce').mean())
                if col in case_df.columns:
                    summary[f'case_{col}'] = float(pd.to_numeric(case_df[col], errors='coerce').mean())
        return broadcast_object(summary, is_ddp=is_ddp, src=0)
    slice_df = pd.DataFrame(rows)
    case_df = summarize_case_metrics(slice_df, args)
    slice_df.to_csv(metrics_dir / f'epoch_{epoch:03d}_{split}_slice.csv', index=False, encoding='utf-8-sig')
    if len(case_df) > 0:
        case_df.insert(1, 'epoch', int(epoch))
        case_df.insert(2, 'split', split)
    case_df.to_csv(metrics_dir / f'epoch_{epoch:03d}_{split}_case.csv', index=False, encoding='utf-8-sig')
    summary = {'loss': total_loss / max(total_n, 1.0)}
    metric_cols = ['mse', 'mae', 'bce', 'kl', 'top_peak_error_px', 'top_hit5', 'top_hit10', 'top_hit20', 'num_gt_peaks', 'num_pred_peaks', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_match_dist_px', 'pred_peak_value', 'gt_peak_value'] + get_froc_metric_cols(args)
    for col in metric_cols:
        if col in slice_df.columns:
            summary[f'slice_{col}'] = float(pd.to_numeric(slice_df[col], errors='coerce').mean())
        if col in case_df.columns:
            summary[f'case_{col}'] = float(pd.to_numeric(case_df[col], errors='coerce').mean())
    return summary

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--base', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--hard_bg_fraction', type=float, default=None)
    parser.add_argument('--hard_bg_min_k', type=int, default=None)
    parser.add_argument('--hard_bg_max_k', type=int, default=None)
    parser.add_argument('--min_peak_pixels', type=int, default=None)
    parser.add_argument('--pred_peak_threshold', type=float, default=0.8)
    parser.add_argument('--gt_peak_threshold', type=float, default=0.99)
    parser.add_argument('--peak_nms_kernel', type=int, default=9)
    parser.add_argument('--match_radius', type=float, default=15.0)
    parser.add_argument('--max_pred_peaks', type=int, default=3)
    parser.add_argument('--max_gt_peaks', type=int, default=3)
    parser.add_argument('--froc_min_peak_score', type=float, default=0.0001)
    parser.add_argument('--max_froc_pred_peaks', type=int, default=50)
    parser.add_argument('--froc_fp_targets', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    parser.add_argument('--monitor_metrics', type=str, nargs='+', default=['case_sens_at_1fp', 'case_sens_at_2fp', 'case_sens_at_0_5fp', 'case_kl', 'case_mean_match_dist_px', 'case_top_hit10'], help='Validation metrics used for multi-metric early stopping. Training stops only when all listed metrics have no improvement for patience epochs. Default: case_sens_at_1fp case_sens_at_2fp case_sens_at_0_5fp case_kl case_mean_match_dist_px case_top_hit10.')
    parser.add_argument('--monitor', type=str, default=None, help='Backward-compatible single primary monitor. If set, it is placed first in monitor_metrics.')
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    parser.add_argument('--run_test_every_epoch', action='store_true', help=argparse.SUPPRESS)
    return parser

def resolve_monitor_metrics(args):
    metrics = [normalize_monitor_name(m) for m in args.monitor_metrics]
    if args.monitor is not None:
        primary = normalize_monitor_name(args.monitor)
        metrics = [primary] + [m for m in metrics if m != primary]
    deduped = []
    for m in metrics:
        if m not in deduped:
            deduped.append(m)
    if len(deduped) == 0:
        raise ValueError('monitor_metrics is empty.')
    args.monitor_metrics = deduped
    args.primary_monitor = deduped[0]
    return args

def build_best_summary_rows(monitor_metrics, best_metrics, best_epochs, bad_epochs_by_metric, best_rows):
    out = []
    for monitor in monitor_metrics:
        best_row = best_rows.get(monitor)
        direction = 'higher' if metric_higher_is_better(monitor) else 'lower'
        row = {'monitor': monitor, 'direction': direction, 'best_epoch': int(best_epochs.get(monitor, 0)), 'bad_epochs': int(bad_epochs_by_metric.get(monitor, 0)), 'best_val_value': float(best_metrics.get(monitor, np.nan))}
        if best_row is not None:
            row[f'best_val_{monitor}'] = float(best_row.get(f'val_{monitor}', np.nan))
            core_metrics = ['case_sens_at_0_5fp', 'case_sens_at_1fp', 'case_sens_at_2fp', 'case_f1', 'case_precision', 'case_recall', 'case_kl', 'case_top_peak_error_px', 'case_mean_match_dist_px', 'case_top_hit10', 'case_mse', 'case_mae', 'case_bce', 'slice_f1', 'slice_precision', 'slice_recall', 'slice_kl', 'slice_top_peak_error_px', 'slice_mean_match_dist_px', 'slice_top_hit10']
            for m in core_metrics:
                row[f'val_{m}'] = best_row.get(f'val_{m}', np.nan)
        out.append(row)
    return out

def main():
    args = build_argparser().parse_args()
    required = {'data_root': args.data_root, 'out_dir': args.out_dir, 'hard_bg_fraction': args.hard_bg_fraction, 'hard_bg_min_k': args.hard_bg_min_k, 'hard_bg_max_k': args.hard_bg_max_k, 'min_peak_pixels': args.min_peak_pixels}
    missing = [k for k, v in required.items() if v in (None, '')]
    if missing:
        raise ValueError('Set required arguments: ' + ', '.join(missing))
    args = resolve_monitor_metrics(args)
    is_ddp, rank, local_rank, world_size = setup_ddp(args)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed + rank)
    if is_main(rank):
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (Path(args.out_dir) / 'metrics').mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()
    train_ds = DeepLesionHeatmapDataset(data_root=args.data_root, split='train')
    val_ds = DeepLesionHeatmapDataset(data_root=args.data_root, split='val')
    train_loader, train_sampler = make_train_loader(train_ds, args.batch_size, args.num_workers, is_ddp, rank, world_size)
    val_loader = make_eval_loader(val_ds, args.eval_batch_size, args.num_workers, is_ddp, rank, world_size)
    model = UNetSECoord(in_ch=1, out_ch=1, base=args.base).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    criterion = HeatmapLoss(hard_bg_fraction=args.hard_bg_fraction, hard_bg_min_k=args.hard_bg_min_k, hard_bg_max_k=args.hard_bg_max_k, min_peak_pixels=args.min_peak_pixels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    amp_enabled = not args.no_amp and device.type == 'cuda'
    scaler = GradScaler(enabled=amp_enabled)
    best_metrics = {m: get_initial_best(m) for m in args.monitor_metrics}
    best_epochs = {m: 0 for m in args.monitor_metrics}
    bad_epochs_by_metric = {m: 0 for m in args.monitor_metrics}
    best_rows: Dict[str, Optional[Dict[str, float]]] = {m: None for m in args.monitor_metrics}
    history: List[Dict[str, float]] = []
    if is_main(rank):
        print(f'[INFO] data_root: {args.data_root}')
        print(f'[INFO] train samples: {len(train_ds)}')
        print(f'[INFO] val samples:   {len(val_ds)}')
        print(f'[INFO] train image root: {train_ds.image_root}')
        print(f'[INFO] train label root: {train_ds.label_root}')
        print(f'[INFO] world_size: {world_size}')
        print(f'[INFO] amp: {amp_enabled}')
        print(f'[INFO] model: UNetSECoord(base={args.base})')
        print(f'[INFO] multi-peak eval: pred_rel_thr={args.pred_peak_threshold}*max_pred, gt_thr={args.gt_peak_threshold}, nms={args.peak_nms_kernel}, radius={args.match_radius}px')
        print(f'[INFO] FROC eval: min_peak_score={args.froc_min_peak_score}, max_froc_pred_peaks={args.max_froc_pred_peaks}, targets={args.froc_fp_targets} FP/image')
        print('[INFO] monitor policy: early stop only when all monitor metrics reach patience.')
        for m in args.monitor_metrics:
            direction = 'higher' if metric_higher_is_better(m) else 'lower'
            print(f'[INFO] monitor: val_{m}, {direction} is better, checkpoint=best_{sanitize_metric_name(m)}.pth')
        print(f'[INFO] primary best.pth follows: val_{args.primary_monitor}')
    for epoch in range(1, args.epochs + 1):
        train_step = train_one_epoch(model, train_loader, train_sampler, optimizer, scaler, criterion, device, epoch, args, is_ddp)
        scheduler.step()
        val_eval = evaluate_split(model, val_loader, criterion, device, epoch, 'val', args.out_dir, args, rank, is_ddp)
        stop_now = False
        if is_main(rank):
            row: Dict[str, float] = {'epoch': int(epoch), 'lr': float(optimizer.param_groups[0]['lr'])}
            for k, v in train_step.items():
                row[f'train_{k}'] = float(v)
            for k, v in val_eval.items():
                row[f'val_{k}'] = float(v)
            history.append(row)
            pd.DataFrame(history).to_csv(Path(args.out_dir) / 'epoch_metrics.csv', index=False, encoding='utf-8-sig')
            ckpt_dir = Path(args.out_dir) / 'checkpoints'
            save_model_only(ckpt_dir / f'epoch_{epoch:03d}.pth', model)
            save_model_only(ckpt_dir / 'latest.pth', model)
            for monitor in args.monitor_metrics:
                monitor_key = f'val_{monitor}'
                if monitor_key not in row:
                    raise KeyError(f'Monitor key {monitor_key} not found. Available keys: {list(row.keys())}')
                current = float(row[monitor_key])
                improved = is_improved(current, best_metrics[monitor], monitor)
                if improved:
                    best_metrics[monitor] = current
                    best_epochs[monitor] = epoch
                    bad_epochs_by_metric[monitor] = 0
                    best_rows[monitor] = dict(row)
                    safe_name = sanitize_metric_name(monitor)
                    save_model_only(ckpt_dir / f'best_{safe_name}.pth', model)
                    if monitor == args.primary_monitor:
                        save_model_only(ckpt_dir / 'best.pth', model)
                elif epoch > 20:
                    bad_epochs_by_metric[monitor] += 1
            best_summary_rows = build_best_summary_rows(args.monitor_metrics, best_metrics, best_epochs, bad_epochs_by_metric, best_rows)
            pd.DataFrame(best_summary_rows).to_csv(Path(args.out_dir) / 'best_monitor_summary.csv', index=False, encoding='utf-8-sig')
            monitor_status = ' | '.join([f"{m}: val={fmt_float(row.get(f'val_{m}', np.nan), 5)}, best={fmt_float(best_metrics[m], 5)}@{best_epochs[m]}, bad={bad_epochs_by_metric[m]}/{args.patience}" for m in args.monitor_metrics])
            print(f"Epoch {epoch:03d} | train_loss={fmt_float(row.get('train_step_loss', np.nan), 5)} | val_loss={fmt_float(row.get('val_loss', np.nan), 5)} | {monitor_status}")
            stop_now = all((bad_epochs_by_metric[m] >= args.patience for m in args.monitor_metrics))
        stop_now = broadcast_object(stop_now, is_ddp=is_ddp, src=0)
        if stop_now:
            if is_main(rank):
                print('[EARLY STOP] all monitor metrics reached patience.')
                for m in args.monitor_metrics:
                    print(f'[EARLY STOP] {m}: best_epoch={best_epochs[m]}, best_val={best_metrics[m]:.6f}, bad={bad_epochs_by_metric[m]}/{args.patience}')
            break
    cleanup_ddp(is_ddp)
if __name__ == '__main__':
    main()
