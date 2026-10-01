import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from datasets import DeepLesionHeatmapDataset, heatmap_collate
from models import UNetSECoord
STATE_DICT_KEYS: Sequence[str] = ('model', 'state_dict', 'model_state_dict', 'student')
PREFIX_CANDIDATES: Sequence[str] = ('module.', 'model.', 'student.', 'segmentor.', 'seg.', 'net.')

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Load baseline UNetSECoord weights and save predicted heatmaps.')
    parser.add_argument('--data_root', type=str, default='', help='Dataset root containing test/images and test/labels.')
    parser.add_argument('--checkpoint', type=str, default='', help='Baseline model checkpoint.')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory for predicted heatmaps.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--base', type=int, default=64, help='Must match the training --base value.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda', help='Device specification such as auto, cpu, cuda, or cuda:0.')
    parser.add_argument('--save_mode', type=str, default='both', choices=['raw', 'normalized', 'both'], help='raw saves sigmoid probabilities, normalized rescales each heatmap by its maximum, and both saves both representations.')
    parser.add_argument('--max_samples', type=int, default=0, help='Zero processes all samples; a positive value processes the first N samples.')
    parser.add_argument('--no_amp', action='store_true', help='Disable CUDA AMP.')
    return parser

def resolve_device(device_name: str) -> torch.device:
    device_name = str(device_name).strip().lower()
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False.')
    return device

def amp_context(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16)
    return torch.cuda.amp.autocast(enabled=True)

def extract_state_dict(checkpoint: Any, preferred_keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in preferred_keys:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict) or len(checkpoint) == 0:
        raise RuntimeError('Could not extract a valid state_dict from the checkpoint.')
    if not all((isinstance(key, str) for key in checkpoint.keys())):
        raise RuntimeError('The state_dict contains a non-string parameter name.')
    return checkpoint

def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if state_dict and all((key.startswith(prefix) for key in state_dict)):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict

def load_model_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Model checkpoint not found：{checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint, preferred_keys=STATE_DICT_KEYS)
    for prefix in PREFIX_CANDIDATES:
        state_dict = strip_prefix_if_present(state_dict, prefix)
    model.load_state_dict(state_dict, strict=True)

def probability_to_uint16(probability: np.ndarray) -> np.ndarray:
    probability = np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)
    probability = np.clip(probability, 0.0, 1.0)
    return np.rint(probability * 65535.0).astype(np.uint16)

def probability_to_normalized_uint8(probability: np.ndarray) -> np.ndarray:
    probability = np.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0)
    probability = np.clip(probability, 0.0, None)
    max_value = float(probability.max())
    if max_value > 0.0:
        probability = probability / max_value
    else:
        probability = np.zeros_like(probability, dtype=np.float32)
    probability = np.clip(probability, 0.0, 1.0)
    return np.rint(probability * 255.0).astype(np.uint8)

def save_heatmap(probability: np.ndarray, relative_path: Path, output_root: Path, save_mode: str) -> None:
    relative_path = relative_path.with_suffix('.png')
    if save_mode in {'raw', 'both'}:
        raw_root = output_root / 'raw' if save_mode == 'both' else output_root
        raw_path = raw_root / relative_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_image = probability_to_uint16(probability)
        Image.fromarray(raw_image).save(raw_path)
    if save_mode in {'normalized', 'both'}:
        normalized_root = output_root / 'normalized' if save_mode == 'both' else output_root
        normalized_path = normalized_root / relative_path
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_image = probability_to_normalized_uint8(probability)
        Image.fromarray(normalized_image).save(normalized_path)

def main() -> None:
    args = build_argparser().parse_args()
    device = resolve_device(args.device)
    dataset = DeepLesionHeatmapDataset(data_root=args.data_root, split=args.split)
    image_root = Path(dataset.image_root)
    if args.max_samples > 0:
        num_samples = min(int(args.max_samples), len(dataset))
        inference_dataset = Subset(dataset, range(num_samples))
    else:
        inference_dataset = dataset
    loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda', collate_fn=heatmap_collate, drop_last=False)
    model = UNetSECoord(in_ch=1, out_ch=1, base=args.base).to(device)
    load_model_weights(model=model, checkpoint_path=Path(args.checkpoint), device=device)
    model.eval()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    amp_enabled = not args.no_amp and device.type == 'cuda'
    total_samples = len(inference_dataset)
    saved_samples = 0
    print(f'[INFO] device: {device}')
    print(f'[INFO] split: {args.split}')
    print(f'[INFO] samples: {total_samples}')
    print(f'[INFO] checkpoint: {args.checkpoint}')
    print(f'[INFO] output_dir: {output_root}')
    print(f'[INFO] save_mode: {args.save_mode}')
    with torch.inference_mode():
        for images, _, metas in loader:
            images = images.to(device, non_blocking=True)
            with amp_context(amp_enabled):
                logits = model(images)
                probabilities = torch.sigmoid(logits)
            probabilities_np = probabilities[:, 0].float().cpu().numpy()
            for probability, meta in zip(probabilities_np, metas):
                image_path = Path(meta['image_path'])
                try:
                    relative_path = image_path.relative_to(image_root)
                except ValueError:
                    relative_path = Path(meta['file_name'])
                save_heatmap(probability=probability, relative_path=relative_path, output_root=output_root, save_mode=args.save_mode)
                saved_samples += 1
            print(f'\r[TEST] saved {saved_samples}/{total_samples}', end='', flush=True)
    print()
    print(f'[DONE] Heatmaps saved to: {output_root.resolve()}')
if __name__ == '__main__':
    main()
