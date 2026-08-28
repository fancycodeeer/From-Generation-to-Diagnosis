import argparse
import copy
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
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
import torchvision.transforms.functional as TF
from datasets import DeepLesionHeatmapDataset, heatmap_collate
from losses import HeatmapLoss
from models import UNetSECoord
from models_ import NLayerDiscriminator, ResnetGenerator, get_norm_layer, init_weights
G_NGF = 64
G_N_BLOCKS = 9
G_NORM = 'instance'
TEACHER_INPUT_MODE = 'fake_fake'
STUDENT_INPUT_MODE = 'real_fake'

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

class TargetImageDataset(Dataset):

    def __init__(self, root: str):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f'Target image root does not exist: {self.root}')
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        self.paths = sorted((path for path in self.root.rglob('*') if path.is_file() and path.suffix.lower() in extensions))
        if not self.paths:
            raise RuntimeError(f'No target-domain images found under: {self.root}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert('L')
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, mean=(0.5,), std=(0.5,))
        return tensor

class GANLoss(nn.Module):

    def __init__(self, mode: str):
        super().__init__()
        if mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f'Unsupported GAN mode: {mode}')

    def forward(self, prediction: torch.Tensor, target_is_real: bool):
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)

def set_requires_grad(model: nn.Module, enabled: bool):
    for parameter in model.parameters():
        parameter.requires_grad_(enabled)

def next_target_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return (batch, iterator)

def is_main(rank: int) -> bool:
    return rank == 0

def get_raw_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

def all_reduce_sum(x: float, device: torch.device, is_ddp: bool) -> float:
    t = torch.tensor([float(x)], dtype=torch.float64, device=device)
    if is_ddp:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())

def broadcast_object(obj: Any, is_ddp: bool, src: int=0) -> Any:
    if not is_ddp:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]

class DistributedEvalSampler(Sampler):

    def __init__(self, dataset, num_replicas: Optional[int]=None, rank: Optional[int]=None):
        self.dataset = dataset
        self.num_replicas = dist.get_world_size() if num_replicas is None else int(num_replicas)
        self.rank = dist.get_rank() if rank is None else int(rank)
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
    raw_model = get_raw_model(model)
    torch.save(raw_model.state_dict(), path)

def load_model_only(path: Path, model, device: torch.device):
    raw_model = get_raw_model(model)
    state = torch.load(path, map_location=device)
    raw_model.load_state_dict(state, strict=True)

def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if len(state_dict) == 0:
        return state_dict
    if all((k.startswith(prefix) for k in state_dict.keys())):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def extract_state_dict(ckpt: Any, preferred_keys: List[str]):
    if isinstance(ckpt, dict):
        for key in preferred_keys:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt

def load_state_dict_flexible(model: nn.Module, ckpt_path: str, device: torch.device, strict: bool=True, preferred_keys: Optional[List[str]]=None, prefix_candidates: Optional[List[str]]=None) -> Tuple[List[str], List[str]]:
    if not ckpt_path:
        raise ValueError('ckpt_path is empty.')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    if preferred_keys is None:
        preferred_keys = ['model', 'state_dict', 'model_state_dict']
    state_dict = extract_state_dict(ckpt, preferred_keys)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f'Could not extract state_dict from checkpoint: {ckpt_path}')
    if prefix_candidates is None:
        prefix_candidates = ['module.', 'model.', 'teacher.', 'student.', 'segmentor.', 'seg.', 'net.', 'netG_A.', 'G_A.', 'translator.', 'generator.', 'netG.', 'G.']
    for prefix in prefix_candidates:
        state_dict = strip_prefix_if_present(state_dict, prefix)
    raw_model = get_raw_model(model)
    incompatible = raw_model.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(incompatible, 'missing_keys', []))
    unexpected = list(getattr(incompatible, 'unexpected_keys', []))
    return (missing, unexpected)

def save_translator_only(path: Path, translator: nn.Module):
    raw = get_raw_model(translator)
    torch.save(raw.state_dict(), path)

def load_translator_only(path: Path, translator: nn.Module, device: torch.device):
    raw = get_raw_model(translator)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'translator' in ckpt and isinstance(ckpt['translator'], dict):
        state = ckpt['translator']
    else:
        state = ckpt
    if not isinstance(state, dict):
        raise RuntimeError(f'Could not extract state_dict from translator checkpoint: {path}')
    for prefix in ['module.', 'translator.', 'generator.', 'netG.', 'G.', 'netG_A.', 'G_A.']:
        state = strip_prefix_if_present(state, prefix)
    raw.load_state_dict(state, strict=True)

def save_joint_checkpoint(path: Path, student: nn.Module, teacher: nn.Module, translator: nn.Module, discriminator: nn.Module, epoch: int, args, extra: Optional[Dict[str, Any]]=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {'epoch': int(epoch), 'teacher_input_mode': TEACHER_INPUT_MODE, 'student_input_mode': STUDENT_INPUT_MODE, 'student': get_raw_model(student).state_dict(), 'teacher': get_raw_model(teacher).state_dict(), 'translator': get_raw_model(translator).state_dict(), 'discriminator': get_raw_model(discriminator).state_dict()}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)

def build_translator(args, device: torch.device):
    norm_layer = get_norm_layer(G_NORM)
    translator = ResnetGenerator(input_nc=1, output_nc=1, ngf=G_NGF, norm_layer=norm_layer, use_dropout=False, n_blocks=G_N_BLOCKS).to(device)
    missing, unexpected = load_state_dict_flexible(translator, ckpt_path=args.translator_ckpt, device=device, strict=args.translator_strict, preferred_keys=['G_A', 'netG_A', 'translator', 'generator', 'state_dict', 'model', 'model_state_dict', 'netG', 'G'], prefix_candidates=['module.', 'netG_A.', 'G_A.', 'translator.', 'generator.', 'netG.', 'G.'])
    return (translator, missing, unexpected)

def build_detector(args) -> nn.Module:
    return UNetSECoord(in_ch=2, out_ch=1, base=args.base)

def real_to_g_input(real_01: torch.Tensor) -> torch.Tensor:
    return real_01.float().clamp(0.0, 1.0) * 2.0 - 1.0

def g_output_to_detector_input(fake_m11: torch.Tensor) -> torch.Tensor:
    return ((fake_m11 + 1.0) * 0.5).clamp(0.0, 1.0)

def unwrap_generator_output(fake):
    if isinstance(fake, (tuple, list)):
        fake = fake[0]
    if isinstance(fake, dict):
        for key in ['fake', 'fake_B', 'out', 'output', 'image']:
            if key in fake:
                fake = fake[key]
                break
        if isinstance(fake, dict):
            raise RuntimeError('Generator returned a dictionary without a supported output key.')
    return fake

def generate_fake_modality(real_01: torch.Tensor, translator: nn.Module) -> torch.Tensor:
    fake = translator(real_to_g_input(real_01))
    fake = unwrap_generator_output(fake)
    if fake.dim() != 4:
        raise RuntimeError(f'Generator output must be BCHW，actual shape={tuple(fake.shape)}')
    if fake.shape[-2:] != real_01.shape[-2:]:
        fake = F.interpolate(fake, size=real_01.shape[-2:], mode='bilinear', align_corners=False)
    return torch.clamp(fake, -1.0, 1.0)

def build_teacher_input(fake_m11: torch.Tensor) -> torch.Tensor:
    fake_01 = g_output_to_detector_input(fake_m11)
    return torch.cat([fake_01, fake_01], dim=1)

def build_student_input(real_01: torch.Tensor, fake_m11: torch.Tensor) -> torch.Tensor:
    fake_01 = g_output_to_detector_input(fake_m11)
    real_01 = real_01.float().clamp(0.0, 1.0)
    return torch.cat([real_01, fake_01], dim=1)

def print_range_once(args, rank: int, real_01: torch.Tensor, fake_m11: torch.Tensor, student_input: torch.Tensor, teacher_input: torch.Tensor, hms: torch.Tensor):
    if not is_main(rank):
        return
    if getattr(args, '_range_printed', False):
        return
    args._range_printed = True
    print(f'[RangeCheck] real_01=({real_01.min().item():.4f},{real_01.max().item():.4f}) | fake_m11=({fake_m11.min().item():.4f},{fake_m11.max().item():.4f}) | student_input=({student_input.min().item():.4f},{student_input.max().item():.4f}) | teacher_input=({teacher_input.min().item():.4f},{teacher_input.max().item():.4f}) | heatmap=({hms.min().item():.4f},{hms.max().item():.4f})')
    print('[RangeCheck expected] dataset real=[0,1], G input=[-1,1], G output=[-1,1], detector input=[0,1].')

class StudentEvalWrapper(nn.Module):

    def __init__(self, student: nn.Module, translator: nn.Module):
        super().__init__()
        self.student = student
        self.translator = translator

    def forward(self, real_01: torch.Tensor) -> torch.Tensor:
        fake = generate_fake_modality(real_01, self.translator)
        x = build_student_input(real_01, fake)
        return self.student(x)

class FeatureHooker:

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = get_raw_model(model)
        self.layer_names = list(layer_names)
        if not self.layer_names:
            raise ValueError('feature_layers must contain at least one exact module name')
        modules = dict(self.model.named_modules())
        missing = [name for name in self.layer_names if name not in modules]
        if missing:
            raise ValueError('Unknown feature layer names: ' + ', '.join(missing))
        self.enabled = True
        self.features: List[torch.Tensor] = []
        self.handles = []
        self.selected_names = self.layer_names
        for name in self.layer_names:
            module = modules[name]

            def hook(_module, _inputs, output):
                if not self.enabled:
                    return
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if isinstance(output, torch.Tensor) and output.dim() == 4:
                    self.features.append(output)
            self.handles.append(module.register_forward_hook(hook))

    def clear(self):
        self.features = []

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

def kd_weight_ramp(epoch: int, args) -> float:
    if args.distill_mode == 'none':
        return 0.0
    if epoch <= args.kd_warmup_epochs:
        return 0.0
    if args.kd_ramp_epochs <= 0:
        return 1.0
    t = (float(epoch) - float(args.kd_warmup_epochs)) / float(args.kd_ramp_epochs)
    return float(max(0.0, min(1.0, t)))

@torch.no_grad()
def per_sample_top_peak_dist(prob: torch.Tensor, gt: torch.Tensor, eps: float=1e-08) -> torch.Tensor:
    b, _, h, w = prob.shape
    p = prob[:, 0].reshape(b, -1)
    g = gt[:, 0].reshape(b, -1)
    pred_idx = p.argmax(dim=1)
    gt_idx = g.argmax(dim=1)
    pred_y = torch.div(pred_idx, w, rounding_mode='floor').float()
    pred_x = (pred_idx % w).float()
    gt_y = torch.div(gt_idx, w, rounding_mode='floor').float()
    gt_x = (gt_idx % w).float()
    dist = torch.sqrt((pred_x - gt_x).pow(2) + (pred_y - gt_y).pow(2))
    dist = dist / float((h ** 2 + w ** 2) ** 0.5)
    valid = g.sum(dim=1) > eps
    dist = torch.where(valid, dist, torch.zeros_like(dist))
    return dist

def per_sample_heatmap_risk(logits: torch.Tensor, gt: torch.Tensor, args, eps: float=1e-08) -> torch.Tensor:
    logits = logits.float()
    gt = gt.float().clamp(0.0, 1.0)
    prob = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction='none').flatten(1).mean(dim=1)
    mse = (prob - gt).pow(2).flatten(1).mean(dim=1)
    p = prob.flatten(1)
    g = gt.flatten(1)
    valid = g.sum(dim=1) > eps
    kl = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
    if valid.any():
        p_valid = p[valid]
        g_valid = g[valid]
        p_dist = p_valid / p_valid.sum(dim=1, keepdim=True).clamp_min(eps)
        g_dist = g_valid / g_valid.sum(dim=1, keepdim=True).clamp_min(eps)
        kl[valid] = (g_dist * (torch.log(g_dist + eps) - torch.log(p_dist + eps))).sum(dim=1)
    peak_dist = per_sample_top_peak_dist(prob, gt, eps=eps).to(dtype=logits.dtype, device=logits.device)
    risk = bce + float(args.sfd_risk_kl_weight) * kl + float(args.sfd_risk_mse_weight) * mse + float(args.sfd_risk_peak_weight) * peak_dist
    return risk.detach()

def build_sfd_gate(student_risk: torch.Tensor, teacher_risk: torch.Tensor, args) -> torch.Tensor:
    if args.distill_mode == 'none':
        return torch.zeros_like(student_risk.detach())
    if args.distill_mode == 'fd':
        return torch.ones_like(student_risk.detach())
    return (student_risk.detach() - teacher_risk.detach() > float(args.sfd_margin)).float()

def normalize_attention(attn: torch.Tensor, eps: float=1e-06) -> torch.Tensor:
    b = attn.shape[0]
    flat = attn.flatten(1)
    amin = flat.min(dim=1).values.view(b, 1, 1, 1)
    amax = flat.max(dim=1).values.view(b, 1, 1, 1)
    return (attn - amin) / (amax - amin + eps)

def heatmap_roi_weight(gt: torch.Tensor, teacher_feat: torch.Tensor, args) -> torch.Tensor:
    gt_small = F.interpolate(gt.float(), size=teacher_feat.shape[-2:], mode='bilinear', align_corners=False)
    gt_small = gt_small.clamp(0.0, 1.0)
    roi = float(args.kd_bg_weight) + float(args.kd_fg_weight) * gt_small
    if args.kd_peak_weight > 0:
        b = gt_small.shape[0]
        flat = gt_small.flatten(1)
        maxv = flat.max(dim=1).values.view(b, 1, 1, 1).clamp_min(1e-06)
        peak = (gt_small >= maxv * float(args.kd_peak_rel_thr)).float()
        roi = roi + float(args.kd_peak_weight) * peak
    if args.kd_teacher_attn_weight > 0:
        with torch.no_grad():
            attn = teacher_feat.detach().pow(2).mean(dim=1, keepdim=True)
            attn = normalize_attention(attn)
        roi = roi * (1.0 + float(args.kd_teacher_attn_weight) * attn)
    return roi.clamp_min(0.0)

def masked_feature_distill_loss(student_feats: List[torch.Tensor], teacher_feats: List[torch.Tensor], gt: torch.Tensor, gate: torch.Tensor, args) -> Optional[torch.Tensor]:
    if len(student_feats) == 0 or len(teacher_feats) == 0:
        return None
    n = min(len(student_feats), len(teacher_feats))
    student_feats = student_feats[-n:]
    teacher_feats = teacher_feats[-n:]
    gate = gate.view(-1, 1, 1, 1)
    total = None
    used = 0
    eps = 1e-06
    for s, t in zip(student_feats, teacher_feats):
        if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
            continue
        if s.dim() != 4 or t.dim() != 4:
            continue
        if s.shape[1] != t.shape[1]:
            continue
        t = t.detach()
        if t.shape[-2:] != s.shape[-2:]:
            t = F.interpolate(t, size=s.shape[-2:], mode='bilinear', align_corners=False)
        roi = heatmap_roi_weight(gt, t, args)
        weight = roi * gate
        if float(torch.sum(weight).detach().item()) <= eps:
            loss_i = s.sum() * 0.0
        else:
            s_norm = F.normalize(s, p=2, dim=1)
            t_norm = F.normalize(t, p=2, dim=1)
            per = (s_norm - t_norm).pow(2)
            denom = torch.sum(weight) * s.shape[1] + eps
            loss_i = torch.sum(per * weight) / denom
        total = loss_i if total is None else total + loss_i
        used += 1
    if used == 0:
        return None
    return total / float(used)

def zero_loss_like(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0

def train_joint_sfd_one_epoch(student, teacher, translator, discriminator, loader, target_loader, sampler, target_sampler, optimizer_student, optimizer_teacher, optimizer_g, optimizer_d, scaler, criterion, gan_loss, device, epoch: int, args, rank: int, is_ddp: bool, student_hooker: FeatureHooker, teacher_hooker: FeatureHooker):
    student.train()
    teacher.train()
    translator.train()
    discriminator.train()
    student_hooker.enable()
    teacher_hooker.enable()
    if sampler is not None:
        sampler.set_epoch(epoch)
    if target_sampler is not None:
        target_sampler.set_epoch(epoch)
    target_iterator = iter(target_loader)
    amp_enabled = not args.no_amp and device.type == 'cuda'
    ramp = kd_weight_ramp(epoch, args)
    total_loss = 0.0
    total_student_task = 0.0
    total_teacher_task = 0.0
    total_feat_kd = 0.0
    total_trans = 0.0
    total_discriminator = 0.0
    total_gate_mean = 0.0
    total_gate_ratio = 0.0
    total_student_risk = 0.0
    total_teacher_risk = 0.0
    total_student_focal = 0.0
    total_student_mse = 0.0
    total_teacher_focal = 0.0
    total_teacher_mse = 0.0
    total_n = 0.0
    for imgs, hms, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        hms = hms.to(device, non_blocking=True)
        real_target, target_iterator = next_target_batch(target_iterator, target_loader)
        real_target = real_target.to(device, non_blocking=True)
        optimizer_student.zero_grad(set_to_none=True)
        optimizer_teacher.zero_grad(set_to_none=True)
        optimizer_g.zero_grad(set_to_none=True)
        student_hooker.clear()
        teacher_hooker.clear()
        set_requires_grad(discriminator, False)
        with autocast(enabled=amp_enabled):
            fake = generate_fake_modality(imgs, translator)
            teacher_input = build_teacher_input(fake)
            student_input = build_student_input(imgs, fake)
            print_range_once(args, rank, imgs, fake, student_input, teacher_input, hms)
            teacher_logits = teacher(teacher_input)
            student_logits = student(student_input)
            teacher_task, teacher_comps = criterion(teacher_logits, hms)
            student_task, student_comps = criterion(student_logits, hms)
            feat_kd = zero_loss_like(student_logits)
            trans_loss = zero_loss_like(student_logits)
            teacher_risk = per_sample_heatmap_risk(teacher_logits, hms, args)
            student_risk = per_sample_heatmap_risk(student_logits, hms, args)
            gate = build_sfd_gate(student_risk, teacher_risk, args).to(student_logits.dtype)
            if args.distill_mode != 'none' and ramp > 0:
                if args.lambda_feat_kd > 0:
                    kd_feat_value = masked_feature_distill_loss(student_hooker.features, teacher_hooker.features, hms, gate, args)
                    if kd_feat_value is not None:
                        feat_kd = kd_feat_value
            if float(args.lambda_trans) > 0:
                trans_loss = gan_loss(discriminator(fake), True)
            loss = student_task + teacher_task + ramp * float(args.lambda_feat_kd) * feat_kd + float(args.lambda_trans) * trans_loss
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer_student)
            scaler.unscale_(optimizer_teacher)
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(translator.parameters(), args.grad_clip)
        scaler.step(optimizer_student)
        scaler.step(optimizer_teacher)
        scaler.step(optimizer_g)
        set_requires_grad(discriminator, True)
        optimizer_d.zero_grad(set_to_none=True)
        if real_target.shape[-2:] != fake.shape[-2:]:
            real_target = F.interpolate(real_target, size=fake.shape[-2:], mode='bilinear', align_corners=False)
        with autocast(enabled=amp_enabled):
            loss_d_real = gan_loss(discriminator(real_target), True)
            loss_d_fake = gan_loss(discriminator(fake.detach()), False)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
        scaler.scale(loss_d).backward()
        scaler.step(optimizer_d)
        scaler.update()
        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        total_student_task += float(student_task.item()) * bs
        total_teacher_task += float(teacher_task.item()) * bs
        total_feat_kd += float(feat_kd.detach().item()) * bs
        total_trans += float(trans_loss.detach().item()) * bs
        total_discriminator += float(loss_d.detach().item()) * bs
        total_gate_mean += float(gate.detach().mean().item()) * bs
        total_gate_ratio += float((gate.detach() > 0.5).float().mean().item()) * bs
        total_student_risk += float(student_risk.detach().mean().item()) * bs
        total_teacher_risk += float(teacher_risk.detach().mean().item()) * bs
        total_student_focal += float(student_comps.get('focal', torch.tensor(0.0, device=device)).item()) * bs
        total_student_mse += float(student_comps.get('weighted_mse', torch.tensor(0.0, device=device)).item()) * bs
        total_teacher_focal += float(teacher_comps.get('focal', torch.tensor(0.0, device=device)).item()) * bs
        total_teacher_mse += float(teacher_comps.get('weighted_mse', torch.tensor(0.0, device=device)).item()) * bs
        total_n += float(bs)
    student_hooker.disable()
    teacher_hooker.disable()
    reduced = {}
    for key, value in {'loss': total_loss, 'student_task': total_student_task, 'teacher_task': total_teacher_task, 'feat_kd': total_feat_kd, 'trans': total_trans, 'discriminator': total_discriminator, 'gate_mean': total_gate_mean, 'gate_ratio': total_gate_ratio, 'student_risk': total_student_risk, 'teacher_risk': total_teacher_risk, 'student_focal': total_student_focal, 'student_mse': total_student_mse, 'teacher_focal': total_teacher_focal, 'teacher_mse': total_teacher_mse, 'n': total_n}.items():
        reduced[key] = all_reduce_sum(value, device, is_ddp)
    n = max(reduced['n'], 1.0)
    return {'step_loss': reduced['loss'] / n, 'step_task_loss': reduced['student_task'] / n, 'step_teacher_task_loss': reduced['teacher_task'] / n, 'step_focal': reduced['student_focal'] / n, 'step_weighted_mse': reduced['student_mse'] / n, 'step_teacher_focal': reduced['teacher_focal'] / n, 'step_teacher_weighted_mse': reduced['teacher_mse'] / n, 'step_feat_kd': reduced['feat_kd'] / n, 'step_trans': reduced['trans'] / n, 'step_discriminator': reduced['discriminator'] / n, 'step_gate_mean': reduced['gate_mean'] / n, 'step_gate_ratio': reduced['gate_ratio'] / n, 'step_kd_ramp': float(ramp), 'step_student_risk': reduced['student_risk'] / n, 'step_teacher_risk': reduced['teacher_risk'] / n}

@torch.no_grad()
def evaluate_split(model, loader, criterion, device, epoch: int, split: str, out_dir: str, args, rank: int, is_ddp: bool):
    model.eval()
    amp_enabled = not args.no_amp and device.type == 'cuda'
    total_loss, total_n, rows = (0.0, 0.0, [])
    for imgs, hms, metas in loader:
        imgs = imgs.to(device, non_blocking=True)
        hms = hms.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(imgs)
            loss, _ = criterion(logits, hms)
            prob = torch.sigmoid(logits)
        bs = imgs.size(0)
        total_loss += float(loss.item()) * bs
        total_n += float(bs)
        metric_rows = compute_batch_metrics(prob.float(), hms.float(), metas, args)
        for m, meta in zip(metric_rows, metas):
            row = {'epoch': int(epoch), 'split': split, 'case_id': str(meta['case_id']), 'patient_id': str(meta['patient_id']), 'series_id': str(meta.get('series_id', '')), 'slice_id': str(meta.get('slice_id', '')), 'file_name': str(meta['file_name']), 'image_path': str(meta['image_path']), 'heatmap_path': str(meta['heatmap_path'])}
            row.update(m)
            rows.append(row)
    total_loss = all_reduce_sum(total_loss, device, is_ddp)
    total_n = all_reduce_sum(total_n, device, is_ddp)
    metrics_dir = Path(out_dir) / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = ['mse', 'mae', 'bce', 'kl', 'top_peak_error_px', 'top_hit5', 'top_hit10', 'top_hit20', 'num_gt_peaks', 'num_pred_peaks', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_match_dist_px', 'pred_peak_value', 'gt_peak_value'] + get_froc_metric_cols(args)
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
            slice_df = pd.concat(slice_parts, ignore_index=True) if len(slice_parts) > 0 else pd.DataFrame()
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
    for col in metric_cols:
        if col in slice_df.columns:
            summary[f'slice_{col}'] = float(pd.to_numeric(slice_df[col], errors='coerce').mean())
        if col in case_df.columns:
            summary[f'case_{col}'] = float(pd.to_numeric(case_df[col], errors='coerce').mean())
    return summary

def check_patient_leakage(train_ds, val_ds, test_ds, rank: int):
    if not is_main(rank):
        return
    train_p = {str(s['patient_id']) for s in train_ds.samples}
    val_p = {str(s['patient_id']) for s in val_ds.samples}
    test_p = {str(s['patient_id']) for s in test_ds.samples}
    tv = train_p & val_p
    tt = train_p & test_p
    vt = val_p & test_p
    print(f'[CHECK] patient leakage train-val={len(tv)}, train-test={len(tt)}, val-test={len(vt)}')
    if len(tv) or len(tt) or len(vt):
        print('[WARNING] Patient-level leakage detected. Formal metrics are not reliable until split is fixed.')

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
            row[f'corresponding_test_{monitor}'] = float(best_row.get(f'test_{monitor}', np.nan))
            core_metrics = ['case_sens_at_0_5fp', 'case_sens_at_1fp', 'case_sens_at_2fp', 'case_f1', 'case_precision', 'case_recall', 'case_kl', 'case_top_peak_error_px', 'case_mean_match_dist_px', 'case_top_hit10', 'case_mse', 'case_mae', 'case_bce', 'slice_f1', 'slice_precision', 'slice_recall', 'slice_kl', 'slice_top_peak_error_px', 'slice_mean_match_dist_px', 'slice_top_hit10']
            for m in core_metrics:
                row[f'val_{m}'] = best_row.get(f'val_{m}', np.nan)
                row[f'test_{m}'] = best_row.get(f'test_{m}', np.nan)
        out.append(row)
    return out

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
    parser.add_argument('--lr_g', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--weight_decay_g', type=float, default=None)
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
    parser.add_argument('--monitor_metrics', type=str, nargs='+', default=['case_sens_at_1fp', 'case_sens_at_2fp', 'case_sens_at_0_5fp', 'case_kl', 'case_mean_match_dist_px', 'case_top_hit10'])
    parser.add_argument('--monitor', type=str, default=None)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    parser.add_argument('--translator_ckpt', type=str, default='')
    parser.add_argument('--translator_strict', action='store_true', default=True)
    parser.add_argument('--translator_non_strict', dest='translator_strict', action='store_false')
    parser.add_argument('--distill_mode', type=str, default='sfd', choices=['sfd', 'fd', 'none'])
    parser.add_argument('--lambda_feat_kd', type=float, default=0.5)
    parser.add_argument('--kd_warmup_epochs', type=int, default=20)
    parser.add_argument('--kd_ramp_epochs', type=int, default=30)
    parser.add_argument('--sfd_margin', type=float, default=0.02)
    parser.add_argument('--feature_layers', type=str, nargs='+', default=None)
    parser.add_argument('--kd_fg_weight', type=float, default=None)
    parser.add_argument('--kd_bg_weight', type=float, default=None)
    parser.add_argument('--kd_peak_weight', type=float, default=None)
    parser.add_argument('--kd_peak_rel_thr', type=float, default=None)
    parser.add_argument('--kd_teacher_attn_weight', type=float, default=None)
    parser.add_argument('--lambda_trans', type=float, default=0.05)
    parser.add_argument('--translation_target_root', type=str, default='')
    parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['lsgan', 'vanilla'])
    parser.add_argument('--discriminator_ndf', type=int, default=64)
    parser.add_argument('--discriminator_layers', type=int, default=3)
    parser.add_argument('--lr_d', type=float, default=0.0001)
    parser.add_argument('--sfd_risk_kl_weight', type=float, default=0.03)
    parser.add_argument('--sfd_risk_mse_weight', type=float, default=0.05)
    parser.add_argument('--sfd_risk_peak_weight', type=float, default=1.0)
    return parser

def main():
    args = build_argparser().parse_args()
    required = {'data_root': args.data_root, 'out_dir': args.out_dir, 'translator_ckpt': args.translator_ckpt, 'weight_decay_g': args.weight_decay_g, 'hard_bg_fraction': args.hard_bg_fraction, 'hard_bg_min_k': args.hard_bg_min_k, 'hard_bg_max_k': args.hard_bg_max_k, 'min_peak_pixels': args.min_peak_pixels, 'feature_layers': args.feature_layers, 'kd_fg_weight': args.kd_fg_weight, 'kd_bg_weight': args.kd_bg_weight, 'kd_peak_weight': args.kd_peak_weight, 'kd_peak_rel_thr': args.kd_peak_rel_thr, 'kd_teacher_attn_weight': args.kd_teacher_attn_weight, 'translation_target_root': args.translation_target_root}
    missing = [k for k, v in required.items() if v in (None, '', [])]
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
        with open(Path(args.out_dir) / 'config_joint_sfd.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args).copy(), f, indent=2, ensure_ascii=False)
    if is_ddp:
        dist.barrier()
    train_ds = DeepLesionHeatmapDataset(data_root=args.data_root, split='train')
    val_ds = DeepLesionHeatmapDataset(data_root=args.data_root, split='val')
    test_ds = DeepLesionHeatmapDataset(data_root=args.data_root, split='test')
    check_patient_leakage(train_ds, val_ds, test_ds, rank)
    train_loader, train_sampler = make_train_loader(train_ds, args.batch_size, args.num_workers, is_ddp, rank, world_size)
    val_loader = make_eval_loader(val_ds, args.eval_batch_size, args.num_workers, is_ddp, rank, world_size)
    test_loader = make_eval_loader(test_ds, args.eval_batch_size, args.num_workers, is_ddp, rank, world_size)
    student = build_detector(args).to(device)
    teacher = build_detector(args).to(device)
    translator, g_missing, g_unexpected = build_translator(args, device)
    discriminator = NLayerDiscriminator(input_nc=1, ndf=args.discriminator_ndf, n_layers=args.discriminator_layers, norm_layer=get_norm_layer(G_NORM)).to(device)
    init_weights(discriminator, 'normal', 0.02)
    target_dataset = TargetImageDataset(args.translation_target_root)
    target_sampler = None
    if is_ddp:
        target_sampler = DistributedSampler(target_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=target_sampler is None, sampler=target_sampler, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    if is_ddp:
        student = DDP(student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        teacher = DDP(teacher, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        translator = DDP(translator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    student_hooker = FeatureHooker(student, layer_names=args.feature_layers)
    teacher_hooker = FeatureHooker(teacher, layer_names=args.feature_layers)
    student_hooker.disable()
    teacher_hooker.disable()
    criterion = HeatmapLoss(hard_bg_fraction=args.hard_bg_fraction, hard_bg_min_k=args.hard_bg_min_k, hard_bg_max_k=args.hard_bg_max_k, min_peak_pixels=args.min_peak_pixels).to(device)
    optimizer_student = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    optimizer_teacher = torch.optim.AdamW(teacher.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    optimizer_g = torch.optim.AdamW(translator.parameters(), lr=args.lr_g, weight_decay=args.weight_decay_g, betas=(0.9, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    gan_loss = GANLoss(args.gan_mode).to(device)
    scheduler_student = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_student, T_max=args.epochs, eta_min=args.lr * 0.01)
    scheduler_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_teacher, T_max=args.epochs, eta_min=args.lr * 0.01)
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.epochs, eta_min=args.lr_g * 0.01)
    amp_enabled = not args.no_amp and device.type == 'cuda'
    scaler = GradScaler(enabled=amp_enabled)
    best_metrics = {m: get_initial_best(m) for m in args.monitor_metrics}
    best_epochs = {m: 0 for m in args.monitor_metrics}
    bad_epochs_by_metric = {m: 0 for m in args.monitor_metrics}
    best_rows: Dict[str, Optional[Dict[str, float]]] = {m: None for m in args.monitor_metrics}
    history: List[Dict[str, Any]] = []
    if is_main(rank):
        print(f'[INFO] data_root: {args.data_root}')
        print(f'[INFO] train samples: {len(train_ds)}')
        print(f'[INFO] val samples:   {len(val_ds)}')
        print(f'[INFO] test samples:  {len(test_ds)}')
        print(f'[INFO] world_size: {world_size}')
        print(f'[INFO] amp: {amp_enabled}')
        print(f'[INFO] student: UNetSECoord(in_ch=2, base={args.base}), input=[real,fake]')
        print(f'[INFO] teacher: UNetSECoord(in_ch=2, base={args.base}), input=[fake,fake]')
        print(f'[INFO] G: ResnetGenerator(ngf={G_NGF}, n_blocks={G_N_BLOCKS}, norm={G_NORM})')
        print(f'[INFO] G ckpt: {args.translator_ckpt}')
        print(f'[INFO] G missing={g_missing}, unexpected={g_unexpected}')
        print(f'[INFO] KD student feature layers: {student_hooker.selected_names}')
        print(f'[INFO] KD teacher feature layers: {teacher_hooker.selected_names}')
        print(f'[INFO] FROC eval: min_peak_score={args.froc_min_peak_score}, max_froc_pred_peaks={args.max_froc_pred_peaks}, targets={args.froc_fp_targets} FP/image')
        print('[INFO] eval/checkpoint monitor uses student [real,fake].')
        print('[INFO] monitor policy: early stop only when all monitor metrics reach patience.')
        for m in args.monitor_metrics:
            direction = 'higher' if metric_higher_is_better(m) else 'lower'
            print(f'[INFO] monitor: val_{m}, {direction} is better, checkpoint=best_{sanitize_metric_name(m)}.pth')
        print(f'[INFO] primary best.pth follows: val_{args.primary_monitor}')
    for epoch in range(1, args.epochs + 1):
        train_step = train_joint_sfd_one_epoch(student=student, teacher=teacher, translator=translator, discriminator=discriminator, loader=train_loader, target_loader=target_loader, sampler=train_sampler, target_sampler=target_sampler, optimizer_student=optimizer_student, optimizer_teacher=optimizer_teacher, optimizer_g=optimizer_g, optimizer_d=optimizer_d, scaler=scaler, criterion=criterion, gan_loss=gan_loss, device=device, epoch=epoch, args=args, rank=rank, is_ddp=is_ddp, student_hooker=student_hooker, teacher_hooker=teacher_hooker)
        scheduler_student.step()
        scheduler_teacher.step()
        scheduler_g.step()
        eval_wrapper = StudentEvalWrapper(student, translator).to(device)
        val_eval = evaluate_split(eval_wrapper, val_loader, criterion, device, epoch, 'val', args.out_dir, args, rank, is_ddp)
        test_eval = evaluate_split(eval_wrapper, test_loader, criterion, device, epoch, 'test', args.out_dir, args, rank, is_ddp)
        stop_now = False
        if is_main(rank):
            row: Dict[str, Any] = {'epoch': int(epoch), 'lr': float(optimizer_student.param_groups[0]['lr']), 'lr_teacher': float(optimizer_teacher.param_groups[0]['lr']), 'lr_g': float(optimizer_g.param_groups[0]['lr'])}
            for k, v in train_step.items():
                row[f'train_{k}'] = float(v)
            for prefix, summary in [('val', val_eval), ('test', test_eval)]:
                for k, v in summary.items():
                    row[f'{prefix}_{k}'] = float(v)
            history.append(row)
            pd.DataFrame(history).to_csv(Path(args.out_dir) / 'epoch_metrics.csv', index=False, encoding='utf-8-sig')
            ckpt_dir = Path(args.out_dir) / 'checkpoints'
            save_model_only(ckpt_dir / f'epoch_{epoch:03d}.pth', student)
            save_model_only(ckpt_dir / f'teacher_epoch_{epoch:03d}.pth', teacher)
            save_translator_only(ckpt_dir / f'G_epoch_{epoch:03d}.pth', translator)
            save_model_only(ckpt_dir / 'latest.pth', student)
            save_model_only(ckpt_dir / 'latest_teacher.pth', teacher)
            save_translator_only(ckpt_dir / 'latest_G.pth', translator)
            save_joint_checkpoint(ckpt_dir / 'latest_joint.pth', student, teacher, translator, discriminator, epoch, args, extra={'row': row})
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
                    save_model_only(ckpt_dir / f'best_{safe_name}.pth', student)
                    save_model_only(ckpt_dir / f'best_{safe_name}_teacher.pth', teacher)
                    save_translator_only(ckpt_dir / f'best_{safe_name}_G.pth', translator)
                    save_joint_checkpoint(ckpt_dir / f'best_{safe_name}_joint.pth', student, teacher, translator, discriminator, epoch, args, extra={'monitor': monitor, 'best_val_value': current, 'row': row})
                    if monitor == args.primary_monitor:
                        save_model_only(ckpt_dir / 'best.pth', student)
                        save_model_only(ckpt_dir / 'best_teacher.pth', teacher)
                        save_translator_only(ckpt_dir / 'best_G.pth', translator)
                        save_joint_checkpoint(ckpt_dir / 'best_joint.pth', student, teacher, translator, discriminator, epoch, args, extra={'monitor': monitor, 'best_val_value': current, 'row': row})
                elif epoch > 20:
                    bad_epochs_by_metric[monitor] += 1
            best_summary_rows = build_best_summary_rows(args.monitor_metrics, best_metrics, best_epochs, bad_epochs_by_metric, best_rows)
            pd.DataFrame(best_summary_rows).to_csv(Path(args.out_dir) / 'best_monitor_summary.csv', index=False, encoding='utf-8-sig')
            monitor_status = ' | '.join([f"{m}: val={fmt_float(row.get(f'val_{m}', np.nan), 5)}, test={fmt_float(row.get(f'test_{m}', np.nan), 5)}, best={fmt_float(best_metrics[m], 5)}@{best_epochs[m]}, bad={bad_epochs_by_metric[m]}/{args.patience}" for m in args.monitor_metrics])
            sfd_status = f"student_task={fmt_float(row.get('train_step_task_loss', np.nan), 5)} | teacher_task={fmt_float(row.get('train_step_teacher_task_loss', np.nan), 5)} | feat_kd={fmt_float(row.get('train_step_feat_kd', np.nan), 5)} | gate={fmt_float(row.get('train_step_gate_mean', np.nan), 5)} | ramp={fmt_float(row.get('train_step_kd_ramp', np.nan), 3)}"
            print(f"Epoch {epoch:03d} | train_loss={fmt_float(row.get('train_step_loss', np.nan), 5)} | {sfd_status} | val_loss={fmt_float(row.get('val_loss', np.nan), 5)} | test_loss={fmt_float(row.get('test_loss', np.nan), 5)} | {monitor_status}")
            stop_now = all((bad_epochs_by_metric[m] >= args.patience for m in args.monitor_metrics))
        stop_now = broadcast_object(stop_now, is_ddp=is_ddp, src=0)
        if stop_now:
            if is_main(rank):
                print('[EARLY STOP] all monitor metrics reached patience.')
                for m in args.monitor_metrics:
                    print(f'[EARLY STOP] {m}: best_epoch={best_epochs[m]}, best_val={best_metrics[m]:.6f}, bad={bad_epochs_by_metric[m]}/{args.patience}')
            break
    best_epochs = broadcast_object(best_epochs, is_ddp=is_ddp, src=0)
    if is_ddp:
        dist.barrier()
    final_rows = []
    for monitor in args.monitor_metrics:
        safe_name = sanitize_metric_name(monitor)
        best_path = Path(args.out_dir) / 'checkpoints' / f'best_{safe_name}.pth'
        best_g_path = Path(args.out_dir) / 'checkpoints' / f'best_{safe_name}_G.pth'
        exists = best_path.exists() and best_g_path.exists()
        exists = broadcast_object(exists, is_ddp=is_ddp, src=0)
        if not exists:
            if is_main(rank):
                print(f'[WARNING] best student/G not found; final test skipped for monitor={monitor}.')
            continue
        load_model_only(best_path, student, device)
        load_translator_only(best_g_path, translator, device)
        final_wrapper = StudentEvalWrapper(student, translator).to(device)
        split_name = f'test_best_{safe_name}'
        test_summary = evaluate_split(final_wrapper, test_loader, criterion, device, int(best_epochs.get(monitor, 0)), split_name, args.out_dir, args, rank, is_ddp)
        if is_main(rank):
            row = {'monitor': monitor, 'checkpoint': str(best_path), 'g_checkpoint': str(best_g_path), 'best_epoch': int(best_epochs.get(monitor, 0)), 'best_val_value': float(best_metrics.get(monitor, np.nan))}
            for k, v in test_summary.items():
                row[f'test_{k}'] = float(v)
            final_rows.append(row)
            pd.DataFrame([row]).to_csv(Path(args.out_dir) / f'test_metrics_best_{safe_name}.csv', index=False, encoding='utf-8-sig')
            print(f'[FINAL TEST] monitor={monitor}, loaded {best_path.name} + {best_g_path.name}, epoch={best_epochs.get(monitor, 0)}')
            print({'monitor': monitor, 'best_epoch': int(best_epochs.get(monitor, 0)), 'best_val_value': float(best_metrics.get(monitor, np.nan)), 'test_case_sens_at_0_5fp': row.get('test_case_sens_at_0_5fp', float('nan')), 'test_case_sens_at_1fp': row.get('test_case_sens_at_1fp', float('nan')), 'test_case_sens_at_2fp': row.get('test_case_sens_at_2fp', float('nan')), 'test_case_kl': row.get('test_case_kl', float('nan')), 'test_case_mean_match_dist_px': row.get('test_case_mean_match_dist_px', float('nan')), 'test_case_top_hit10': row.get('test_case_top_hit10', float('nan')), 'test_case_f1': row.get('test_case_f1', float('nan')), 'test_case_recall': row.get('test_case_recall', float('nan')), 'test_case_precision': row.get('test_case_precision', float('nan'))})
    if is_main(rank) and len(final_rows) > 0:
        pd.DataFrame(final_rows).to_csv(Path(args.out_dir) / 'test_metrics_best_all_monitors.csv', index=False, encoding='utf-8-sig')
    student_hooker.close()
    teacher_hooker.close()
    cleanup_ddp(is_ddp)
if __name__ == '__main__':
    main()
