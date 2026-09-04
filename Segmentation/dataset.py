import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def list_images(folder: str) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f'Folder not found: {folder}')
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
    files.sort()
    if len(files) == 0:
        raise RuntimeError(f'No images found in folder: {folder}')
    return files

def parse_brats_stem(stem: str) -> Tuple[str, str, str]:
    parts = stem.split('_')
    if len(parts) < 4:
        raise ValueError(f'Unexpected file name format BraTS2021_xxxxx_modality_slice: {stem}')
    slice_id = parts[-1]
    modal = parts[-2]
    case_id = '_'.join(parts[:-2])
    return (case_id, modal, slice_id)

def find_image_by_mask(mask_path: Path, image_dir: Path, modality: str) -> Path:
    case_id, mask_modal, slice_id = parse_brats_stem(mask_path.stem)
    if mask_modal != 'seg':
        raise ValueError(f'Expected seg in the mask filename but received: {mask_path.name}')
    image_stem = f'{case_id}_{modality}_{slice_id}'
    candidate = image_dir / f'{image_stem}{mask_path.suffix}'
    if candidate.exists():
        return candidate
    for ext in IMG_EXTENSIONS:
        candidate = image_dir / f'{image_stem}{ext}'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'No image matches the mask:\n  mask: {mask_path}\n  image_dir: {image_dir}\n  expected_stem: {image_stem}')

class BraTS2021BinarySegDataset(Dataset):

    def __init__(self, dataroot: str, phase: str='train', modality: str='t1', load_size: int=0, crop_size: int=0, no_flip: bool=True):
        super().__init__()
        self.dataroot = Path(dataroot)
        self.phase = phase
        self.modality = modality.lower()
        self.load_size = int(load_size)
        self.crop_size = int(crop_size)
        self.no_flip = bool(no_flip)
        if self.modality not in ['t1', 't1ce', 't2', 'flair']:
            raise ValueError(f'Unsupported modality: {modality}')
        self.mask_dir = self.dataroot / phase / 'mask'
        self.image_dir = self.dataroot / phase / self.modality
        self.mask_paths = list_images(str(self.mask_dir))
        self.pairs: List[Tuple[Path, Path]] = []
        for mask_path in self.mask_paths:
            image_path = find_image_by_mask(mask_path, self.image_dir, self.modality)
            self.pairs.append((image_path, mask_path))
        print(f'[{phase}] Single-modality input: {self.modality}')
        print(f'[{phase}] image_dir: {self.image_dir}')
        print(f'[{phase}] mask_dir : {self.mask_dir}')
        print(f'[{phase}] Number of samples: {len(self.pairs)}')
        if len(self.pairs) > 0:
            print(f'[{phase}] First sample:')
            print(f'  image: {self.pairs[0][0].name}')
            print(f'  mask : {self.pairs[0][1].name}')

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def load_mri(path: Path) -> Image.Image:
        return Image.open(path).convert('L')

    @staticmethod
    def load_mask(path: Path) -> Image.Image:
        return Image.open(path).convert('L')

    @staticmethod
    def image_to_tensor_normalized(img: Image.Image) -> torch.Tensor:
        x = TF.to_tensor(img)
        x = TF.normalize(x, mean=(0.5,), std=(0.5,))
        return x

    @staticmethod
    def mask_to_tensor_binary(mask: Image.Image) -> torch.Tensor:
        arr = np.array(mask, dtype=np.uint8)
        arr = (arr > 0).astype(np.float32)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor

    def paired_transform(self, image: Image.Image, mask: Image.Image):
        if self.load_size > 0:
            image = TF.resize(image, size=[self.load_size, self.load_size], interpolation=TF.InterpolationMode.BICUBIC)
            mask = TF.resize(mask, size=[self.load_size, self.load_size], interpolation=TF.InterpolationMode.NEAREST)
        if self.phase == 'train':
            if self.crop_size > 0:
                i, j, h, w = TF.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
            if not self.no_flip:
                if random.random() < 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
        image = self.image_to_tensor_normalized(image)
        mask = self.mask_to_tensor_binary(mask)
        return (image, mask)

    def __getitem__(self, index: int) -> Dict:
        image_path, mask_path = self.pairs[index]
        image = self.load_mri(image_path)
        mask = self.load_mask(mask_path)
        image, mask = self.paired_transform(image, mask)
        case_id, _, slice_id = parse_brats_stem(mask_path.stem)
        return {'image': image, 'mask': mask, 'image_path': str(image_path), 'mask_path': str(mask_path), 'case_id': case_id, 'slice_id': slice_id}
