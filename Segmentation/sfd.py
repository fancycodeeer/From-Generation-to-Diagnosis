import argparse
import copy
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
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torch.utils.data.distributed import DistributedSampler
from dataset import BraTS2021BinarySegDataset
from loss import ComboSegLoss
from models import UNetCoordinateChannelAttention
from models_ import NLayerDiscriminator, ResnetGenerator, get_norm_layer, init_weights

def parse_args():
    parser = argparse.ArgumentParser('Joint SFD training for BraTS segmentation')
    parser.add_argument('--dataroot', type=str, default='')
    parser.add_argument('--translator_ckpt', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--train_phase', type=str, default='train')
    parser.add_argument('--val_phase', type=str, default='val')
    parser.add_argument('--test_phase', type=str, default='test')
    parser.add_argument('--modality', type=str, default='t1', choices=['t1', 't1ce', 't2', 'flair'])
    parser.add_argument('--load_size', type=int, default=0)
    parser.add_argument('--crop_size', type=int, default=0)
    parser.add_argument('--no_flip', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--min_delta', type=float, default=0.0)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--translator_ngf', type=int, default=64)
    parser.add_argument('--translator_n_blocks', type=int, default=9)
    parser.add_argument('--translator_norm', type=str, default='instance', choices=['instance', 'batch', 'none'])
    parser.add_argument('--feature_layers', nargs='+', default=None)
    parser.add_argument('--lambda_distill', type=float, default=0.5)
    parser.add_argument('--kd_warmup_epochs', type=int, default=20)
    parser.add_argument('--kd_ramp_epochs', type=int, default=30)
    parser.add_argument('--sfd_margin', type=float, default=0.0)
    parser.add_argument('--lambda_trans', type=float, default=0.05)
    parser.add_argument('--translation_target_root', type=str, default='')
    parser.add_argument('--gan_mode', type=str, default='lsgan', choices=['lsgan', 'vanilla'])
    parser.add_argument('--discriminator_ndf', type=int, default=64)
    parser.add_argument('--discriminator_layers', type=int, default=3)
    parser.add_argument('--lr_d', type=float, default=0.0001)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
    args = parser.parse_args()
    required = {'dataroot': args.dataroot, 'translator_ckpt': args.translator_ckpt, 'save_dir': args.save_dir, 'feature_layers': args.feature_layers, 'translation_target_root': args.translation_target_root}
    missing = [key for key, value in required.items() if value in (None, '', [])]
    if missing:
        parser.error('Required settings are empty: ' + ', '.join(missing))
    return args

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

def setup_distributed(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.device = torch.device('cuda', args.local_rank)
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

def is_main(args):
    return args.rank == 0

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def unwrap(model):
    return model.module if isinstance(model, DDP) else model

def strip_module_prefix(state):
    if state and all((str(k).startswith('module.') for k in state)):
        return {str(k)[7:]: v for k, v in state.items()}
    return state

def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ['G_A', 'generator', 'translator', 'model', 'state_dict', 'model_state_dict']:
            value = obj.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(obj, dict) and obj and all((isinstance(k, str) for k in obj)):
        return obj
    raise RuntimeError('Could not extract a translator state_dict from the checkpoint')

def build_translator(args, checkpoint_path: str):
    norm = get_norm_layer(args.translator_norm)
    model = ResnetGenerator(input_nc=1, output_nc=1, ngf=args.translator_ngf, norm_layer=norm, use_dropout=False, n_blocks=args.translator_n_blocks).to(args.device)
    state = extract_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.load_state_dict(strip_module_prefix(state), strict=True)
    return model

def build_segmentor(args):
    return UNetCoordinateChannelAttention(in_channels=2, num_classes=1, base_channels=args.base_channels, bilinear=True, use_attention=True).to(args.device)

class FeatureHooker:

    def __init__(self, model: nn.Module, names: List[str]):
        modules = dict(model.named_modules())
        missing = [name for name in names if name not in modules]
        if missing:
            raise ValueError('Unknown feature layers: ' + ', '.join(missing))
        self.names = list(names)
        self.features: Dict[str, torch.Tensor] = {}
        self.handles = []
        for name in self.names:
            self.handles.append(modules[name].register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):

        def hook(_, __, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise RuntimeError(f'Feature layer {name} did not produce a BCHW tensor')
            self.features[name] = output
        return hook

    def clear(self):
        self.features = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

def generate_fake(images, translator):
    fake = translator(images)
    if isinstance(fake, (tuple, list)):
        fake = fake[0]
    if isinstance(fake, dict):
        for key in ['fake', 'fake_B', 'out', 'output', 'image']:
            if key in fake:
                fake = fake[key]
                break
    if not isinstance(fake, torch.Tensor) or fake.ndim != 4:
        raise RuntimeError('Translator output must be a BCHW tensor')
    if fake.shape[-2:] != images.shape[-2:]:
        fake = F.interpolate(fake, size=images.shape[-2:], mode='bilinear', align_corners=False)
    return fake.clamp(-1.0, 1.0)

def distill_weight(epoch: int, args):
    if epoch <= args.kd_warmup_epochs:
        return 0.0
    if args.kd_ramp_epochs <= 0:
        return float(args.lambda_distill)
    progress = min(1.0, (epoch - args.kd_warmup_epochs) / float(args.kd_ramp_epochs))
    return float(args.lambda_distill) * progress

def hard_gate(student_risk: torch.Tensor, teacher_risk: torch.Tensor, margin: float):
    return (teacher_risk.detach() + float(margin) < student_risk.detach()).float()

def feature_distillation(student_features, teacher_features, gate, names):
    total = None
    for name in names:
        s = student_features[name]
        t = teacher_features[name].detach()
        if s.shape != t.shape:
            raise RuntimeError(f'Feature shape mismatch at {name}: {tuple(s.shape)} vs {tuple(t.shape)}')
        per_sample = (s.float() - t.float()).pow(2).flatten(1).mean(dim=1)
        layer_loss = (per_sample * gate).mean()
        total = layer_loss if total is None else total + layer_loss
    if total is None:
        raise RuntimeError('No feature layer was selected')
    return total

def build_dataset(args, phase, training):
    return BraTS2021BinarySegDataset(dataroot=args.dataroot, phase=phase, modality=args.modality, load_size=args.load_size, crop_size=args.crop_size if training else 0, no_flip=args.no_flip if training else True)

def build_loaders(args):
    train_set = build_dataset(args, args.train_phase, True)
    train_sampler = None
    if args.world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0)
    val_loader = None
    test_loader = None
    if is_main(args):
        val_loader = DataLoader(build_dataset(args, args.val_phase, False), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0)
        test_loader = DataLoader(build_dataset(args, args.test_phase, False), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0)
    return (train_loader, val_loader, test_loader, train_sampler)

def train_epoch(student, teacher, translator, discriminator, loader, target_loader, sampler, target_sampler, optimizer, optimizer_d, scaler, criterion, gan_loss, student_hooks, teacher_hooks, args, epoch):
    student.train()
    teacher.train()
    translator.train()
    discriminator.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    if target_sampler is not None:
        target_sampler.set_epoch(epoch)
    target_iterator = iter(target_loader)
    amp_enabled = args.amp and args.device.type == 'cuda'
    lambda_distill = distill_weight(epoch, args)
    totals = defaultdict(float)
    count = 0
    for batch in loader:
        images = batch['image'].to(args.device, non_blocking=True)
        masks = batch['mask'].to(args.device, non_blocking=True)
        real_target, target_iterator = next_target_batch(target_iterator, target_loader)
        real_target = real_target.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        student_hooks.clear()
        teacher_hooks.clear()
        set_requires_grad(discriminator, False)
        with autocast(enabled=amp_enabled):
            fake = generate_fake(images, translator)
            student_input = torch.cat([images, fake], dim=1)
            teacher_input = torch.cat([fake, fake], dim=1)
            student_logits = student(student_input)
            teacher_logits = teacher(teacher_input)
            student_task, student_parts = criterion(student_logits, masks)
            teacher_task, teacher_parts = criterion(teacher_logits, masks)
            student_risk, _ = criterion.per_sample(student_logits, masks)
            teacher_risk, _ = criterion.per_sample(teacher_logits, masks)
            gate = hard_gate(student_risk, teacher_risk, args.sfd_margin).to(student_logits.dtype)
            distill = feature_distillation(student_hooks.features, teacher_hooks.features, gate, args.feature_layers)
            trans = gan_loss(discriminator(fake), True)
            loss = student_task + teacher_task + lambda_distill * distill + float(args.lambda_trans) * trans
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        set_requires_grad(discriminator, True)
        optimizer_d.zero_grad(set_to_none=True)
        if real_target.shape[-2:] != fake.shape[-2:]:
            real_target = F.interpolate(real_target, size=fake.shape[-2:], mode='bilinear', align_corners=False)
        with autocast(enabled=amp_enabled):
            d_real = gan_loss(discriminator(real_target), True)
            d_fake = gan_loss(discriminator(fake.detach()), False)
            d_loss = 0.5 * (d_real + d_fake)
        scaler.scale(d_loss).backward()
        scaler.step(optimizer_d)
        scaler.update()
        batch_size = images.shape[0]
        count += batch_size
        totals['loss'] += float(loss.detach()) * batch_size
        totals['student_task'] += float(student_task.detach()) * batch_size
        totals['teacher_task'] += float(teacher_task.detach()) * batch_size
        totals['distill'] += float(distill.detach()) * batch_size
        totals['trans'] += float(trans.detach()) * batch_size
        totals['discriminator'] += float(d_loss.detach()) * batch_size
        totals['gate_ratio'] += float(gate.mean().detach()) * batch_size
        totals['student_risk'] += float(student_risk.mean().detach()) * batch_size
        totals['teacher_risk'] += float(teacher_risk.mean().detach()) * batch_size
        for key, value in student_parts.items():
            totals['student_' + key] += float(value.detach()) * batch_size
        for key, value in teacher_parts.items():
            totals['teacher_' + key] += float(value.detach()) * batch_size
    tensor_keys = sorted(totals)
    if args.world_size > 1:
        values = torch.tensor([totals[k] for k in tensor_keys] + [float(count)], dtype=torch.float64, device=args.device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        count = int(values[-1].item())
        totals = {key: float(values[i].item()) for i, key in enumerate(tensor_keys)}
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result['lambda_distill'] = lambda_distill
    return result

def evaluate(model, translator, loader, args):
    model.eval()
    translator.eval()
    cases = defaultdict(lambda: {'tp': 0.0, 'fp': 0.0, 'fn': 0.0})
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(args.device)
            masks = batch['mask'].to(args.device)
            fake = generate_fake(images, translator)
            logits = model(torch.cat([images, fake], dim=1))
            pred = (torch.sigmoid(logits) >= args.threshold).float()
            for i, case_id in enumerate(batch['case_id']):
                p = pred[i]
                g = masks[i]
                cases[str(case_id)]['tp'] += float((p * g).sum().item())
                cases[str(case_id)]['fp'] += float((p * (1.0 - g)).sum().item())
                cases[str(case_id)]['fn'] += float(((1.0 - p) * g).sum().item())
    rows = []
    for case_id, values in sorted(cases.items()):
        tp, fp, fn = (values['tp'], values['fp'], values['fn'])
        precision = tp / max(tp + fp, 1e-08)
        recall = tp / max(tp + fn, 1e-08)
        dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-08)
        iou = tp / max(tp + fp + fn, 1e-08)
        rows.append({'case_id': case_id, 'precision': precision, 'recall': recall, 'dice': dice, 'iou': iou})
    if not rows:
        raise RuntimeError('Evaluation produced no patient-level rows')
    summary = {key: float(np.mean([row[key] for row in rows])) for key in ['precision', 'recall', 'dice', 'iou']}
    return (summary, rows)

def save_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def save_checkpoint(path, epoch, student, teacher, translator, discriminator, optimizer, optimizer_d, scaler, best_dice, args):
    torch.save({'epoch': int(epoch), 'student': unwrap(student).state_dict(), 'teacher': unwrap(teacher).state_dict(), 'translator': unwrap(translator).state_dict(), 'discriminator': unwrap(discriminator).state_dict(), 'optimizer': optimizer.state_dict(), 'optimizer_d': optimizer_d.state_dict(), 'scaler': scaler.state_dict(), 'best_dice': float(best_dice), 'args': vars(args)}, path)

def load_checkpoint(path, student, teacher, translator, discriminator, optimizer, optimizer_d, scaler, device):
    obj = torch.load(path, map_location=device)
    unwrap(student).load_state_dict(obj['student'], strict=True)
    unwrap(teacher).load_state_dict(obj['teacher'], strict=True)
    unwrap(translator).load_state_dict(obj['translator'], strict=True)
    unwrap(discriminator).load_state_dict(obj['discriminator'], strict=True)
    optimizer.load_state_dict(obj['optimizer'])
    optimizer_d.load_state_dict(obj['optimizer_d'])
    scaler.load_state_dict(obj['scaler'])
    return (int(obj['epoch']) + 1, float(obj.get('best_dice', -1.0)))

def main():
    args = setup_distributed(parse_args())
    seed_all(args.seed + args.rank)
    if is_main(args):
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.save_dir) / 'config.json', 'w', encoding='utf-8') as handle:
            json.dump(vars(args), handle, indent=2, default=str)
    train_loader, val_loader, test_loader, train_sampler = build_loaders(args)
    student = build_segmentor(args)
    teacher = build_segmentor(args)
    translator = build_translator(args, args.translator_ckpt)
    norm = get_norm_layer(args.translator_norm)
    discriminator = NLayerDiscriminator(1, ndf=args.discriminator_ndf, n_layers=args.discriminator_layers, norm_layer=norm).to(args.device)
    init_weights(discriminator, 'normal', 0.02)
    target_dataset = TargetImageDataset(args.translation_target_root)
    target_sampler = None
    if args.world_size > 1:
        target_sampler = DistributedSampler(target_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=target_sampler is None, sampler=target_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0, drop_last=False)
    student_hooks = FeatureHooker(student, args.feature_layers)
    teacher_hooks = FeatureHooker(teacher, args.feature_layers)
    if args.world_size > 1:
        student = DDP(student, device_ids=[args.local_rank], output_device=args.local_rank)
        teacher = DDP(teacher, device_ids=[args.local_rank], output_device=args.local_rank)
        translator = DDP(translator, device_ids=[args.local_rank], output_device=args.local_rank)
        discriminator = DDP(discriminator, device_ids=[args.local_rank], output_device=args.local_rank)
    criterion = ComboSegLoss().to(args.device)
    optimizer = torch.optim.Adam(list(student.parameters()) + list(teacher.parameters()) + list(translator.parameters()), lr=args.lr, betas=(0.9, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    gan_loss = GANLoss(args.gan_mode).to(args.device)
    scaler = GradScaler(enabled=args.amp and args.device.type == 'cuda')
    start_epoch = 1
    best_dice = -1.0
    bad_epochs = 0
    if args.resume:
        start_epoch, best_dice = load_checkpoint(args.resume, student, teacher, translator, discriminator, optimizer, optimizer_d, scaler, args.device)
    metrics_path = Path(args.save_dir) / 'metrics.csv'
    if is_main(args) and (not metrics_path.exists()):
        metrics_path.write_text('', encoding='utf-8')
    for epoch in range(start_epoch, args.epochs + 1):
        train_summary = train_epoch(student, teacher, translator, discriminator, train_loader, target_loader, train_sampler, target_sampler, optimizer, optimizer_d, scaler, criterion, gan_loss, student_hooks, teacher_hooks, args, epoch)
        if args.world_size > 1:
            dist.barrier()
        stop = False
        if is_main(args):
            val_summary, val_rows = evaluate(unwrap(student), unwrap(translator), val_loader, args)
            save_rows(Path(args.save_dir) / 'case_metrics' / f'val_epoch_{epoch:03d}.csv', val_rows)
            row = {'epoch': epoch, **{f'train_{k}': v for k, v in train_summary.items()}, **{f'val_{k}': v for k, v in val_summary.items()}}
            exists = metrics_path.stat().st_size > 0
            with open(metrics_path, 'a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            save_checkpoint(Path(args.save_dir) / 'latest.pth', epoch, student, teacher, translator, discriminator, optimizer, optimizer_d, scaler, best_dice, args)
            if val_summary['dice'] > best_dice + args.min_delta:
                best_dice = val_summary['dice']
                bad_epochs = 0
                save_checkpoint(Path(args.save_dir) / 'best.pth', epoch, student, teacher, translator, discriminator, optimizer, optimizer_d, scaler, best_dice, args)
                torch.save(unwrap(student).state_dict(), Path(args.save_dir) / 'best_student.pth')
                torch.save(unwrap(translator).state_dict(), Path(args.save_dir) / 'best_translator.pth')
            else:
                bad_epochs += 1
            stop = bad_epochs >= args.patience
            print(f"epoch={epoch} val_dice={val_summary['dice']:.6f} val_iou={val_summary['iou']:.6f} gate={train_summary['gate_ratio']:.4f}")
        if args.world_size > 1:
            flag = torch.tensor([1 if stop else 0], device=args.device)
            dist.broadcast(flag, src=0)
            stop = bool(flag.item())
        if stop:
            break
    if args.world_size > 1:
        dist.barrier()
    if is_main(args):
        best_path = Path(args.save_dir) / 'best.pth'
        if best_path.exists():
            obj = torch.load(best_path, map_location=args.device)
            unwrap(student).load_state_dict(obj['student'], strict=True)
            unwrap(translator).load_state_dict(obj['translator'], strict=True)
        test_summary, test_rows = evaluate(unwrap(student), unwrap(translator), test_loader, args)
        save_rows(Path(args.save_dir) / 'case_metrics' / 'test.csv', test_rows)
        with open(Path(args.save_dir) / 'test_summary.json', 'w', encoding='utf-8') as handle:
            json.dump(test_summary, handle, indent=2)
    student_hooks.close()
    teacher_hooks.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
if __name__ == '__main__':
    main()
