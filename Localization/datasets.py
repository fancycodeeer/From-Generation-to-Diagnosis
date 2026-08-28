from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def _resample_bilinear():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR

def parse_deeplesion_name(file_name: str) -> Dict[str, str]:
    name = Path(str(file_name)).name
    stem = Path(name).stem
    parts = stem.split('_')
    if len(parts) == 4:
        try:
            patient = str(int(parts[0])).zfill(6)
            study = str(int(parts[1])).zfill(2)
            series = str(int(parts[2])).zfill(2)
            slice_id = str(int(parts[3])).zfill(3)
            series_id = f'{patient}_{study}_{series}'
            canonical = f'{patient}_{study}_{series}_{slice_id}.png'
            return {'file_name': canonical, 'patient_id': patient, 'series_id': series_id, 'slice_id': slice_id, 'case_id': patient}
        except Exception:
            pass
    patient_id = stem.split('_')[0]
    return {'file_name': name, 'patient_id': patient_id, 'series_id': '', 'slice_id': stem, 'case_id': patient_id}

def read_gray_float01(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32)
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        arr = arr.astype(np.float32)
    if arr.size == 0:
        raise RuntimeError(f'Empty image: {path}')
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise RuntimeError(f'Non-finite image values: {path}')
    if vmax <= 1.0 and vmin >= 0.0:
        out = arr
    elif vmax <= 255.0 and vmin >= 0.0:
        out = arr / 255.0
    elif vmax > vmin:
        out = (arr - vmin) / (vmax - vmin)
    else:
        out = np.zeros_like(arr, dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def resize_to_512(arr: np.ndarray, is_label: bool) -> np.ndarray:
    if arr.shape == (512, 512):
        return arr.astype(np.float32)
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))
    img = img.resize((512, 512), resample=_resample_bilinear())
    out = np.array(img).astype(np.float32) / 255.0
    if is_label:
        m = float(out.max())
        if m > 0:
            out = out / m
    return np.clip(out, 0.0, 1.0).astype(np.float32)

class DeepLesionHeatmapDataset(Dataset):

    def __init__(self, data_root: str, split: str):
        self.data_root = Path(data_root)
        self.split = str(split)
        self.image_root = self.data_root / self.split / 'images'
        self.label_root = self.data_root / self.split / 'labels'
        if not self.image_root.exists():
            raise FileNotFoundError(f'image_root not found: {self.image_root}')
        if not self.label_root.exists():
            raise FileNotFoundError(f'label_root not found: {self.label_root}')
        label_paths = sorted((p for p in self.label_root.rglob('*') if p.is_file() and p.suffix.lower() in IMG_EXTS))
        if len(label_paths) == 0:
            raise RuntimeError(f'No label images found in: {self.label_root}')
        self.samples: List[Dict[str, Any]] = []
        missing = []
        for label_path in label_paths:
            rel = label_path.relative_to(self.label_root)
            image_path = self.image_root / rel
            if not image_path.exists():
                missing.append(str(rel))
                continue
            meta = parse_deeplesion_name(label_path.name)
            self.samples.append({'file_name': meta['file_name'], 'case_id': meta['case_id'], 'patient_id': meta['patient_id'], 'series_id': meta['series_id'], 'slice_id': meta['slice_id'], 'image_path': str(image_path), 'heatmap_path': str(label_path)})
        if len(missing) > 0:
            raise RuntimeError(f'{len(missing)} labels have no matched image. Required path format: data_root/{self.split}/images/<same_relative_path_as_label>. First missing: {missing[:10]}')
        if len(self.samples) == 0:
            raise RuntimeError(f'No valid image/label pairs found for split={self.split}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = read_gray_float01(Path(s['image_path']))
        hm = read_gray_float01(Path(s['heatmap_path']))
        img = resize_to_512(img, is_label=False)
        hm = resize_to_512(hm, is_label=True)
        img = torch.from_numpy(img).unsqueeze(0)
        hm = torch.from_numpy(hm).unsqueeze(0)
        return (img, hm, dict(s))

def heatmap_collate(batch):
    imgs, hms, metas = zip(*batch)
    return (torch.stack(imgs, dim=0), torch.stack(hms, dim=0), list(metas))
