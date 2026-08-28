import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def list_images(folder: str) -> List[str]:
    path = Path(folder)
    if not path.exists():
        raise FileNotFoundError(f'Folder not found: {folder}')
    files = sorted((str(p) for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS))
    if not files:
        raise RuntimeError(f'No image found in: {folder}')
    return files

def build_transform(phase: str, input_nc: int=1, load_size: int=0, crop_size: int=0, no_flip: bool=True):
    image_mode = 'L' if input_nc == 1 else 'RGB'
    operations = []
    if load_size > 0:
        operations.append(transforms.Resize((load_size, load_size), interpolation=transforms.InterpolationMode.BICUBIC))
    if phase == 'train' and crop_size > 0:
        operations.append(transforms.RandomCrop(crop_size))
    if phase == 'train' and (not no_flip):
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    operations.extend([transforms.Lambda(lambda image: image.convert(image_mode)), transforms.ToTensor()])
    if input_nc == 1:
        operations.append(transforms.Normalize((0.5,), (0.5,)))
    else:
        operations.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(operations)

def resolve_domains(mode: Optional[str]=None, source_domain: Optional[str]=None, target_domain: Optional[str]=None) -> Tuple[str, str]:
    if source_domain and target_domain:
        return (str(source_domain), str(target_domain))
    mapping = {'t1': ('t1', 't1ce'), 't2': ('t2', 'flair')}
    if mode in mapping:
        return mapping[str(mode)]
    raise ValueError('Provide both source_domain and target_domain for non-BraTS translation tasks')

def _pair_key(path: str, domain: str) -> str:
    stem = Path(path).stem
    domain_tokens = [f'_{domain}_', f'-{domain}-', f'_{domain}-', f'-{domain}_']
    for token in domain_tokens:
        if token in stem:
            return stem.replace(token, '_<DOMAIN>_', 1)
    return stem

class UnpairedBratsDataset(Dataset):

    def __init__(self, dataroot: str, phase: str, transform, mode: Optional[str]=None, source_domain: Optional[str]=None, target_domain: Optional[str]=None):
        super().__init__()
        if phase != 'train':
            raise ValueError("Unpaired training dataset requires phase='train'")
        source_domain, target_domain = resolve_domains(mode, source_domain, target_domain)
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.dir_A = Path(dataroot) / phase / source_domain
        self.dir_B = Path(dataroot) / phase / target_domain
        self.A_paths = list_images(str(self.dir_A))
        self.B_paths = list_images(str(self.dir_B))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = transform

    def __len__(self) -> int:
        return max(self.A_size, self.B_size)

    def __getitem__(self, index: int) -> Dict:
        a_path = self.A_paths[index % self.A_size]
        b_path = self.B_paths[random.randrange(self.B_size)]
        with Image.open(a_path) as image_a:
            a = self.transform(image_a)
        with Image.open(b_path) as image_b:
            b = self.transform(image_b)
        return {'A': a, 'B': b, 'A_path': a_path, 'B_path': b_path}

class PairedBratsValDataset(Dataset):

    def __init__(self, dataroot: str, phase: str, transform, mode: Optional[str]=None, source_domain: Optional[str]=None, target_domain: Optional[str]=None, pairing: str='key'):
        super().__init__()
        if phase not in {'val', 'test'}:
            raise ValueError("Paired evaluation dataset requires phase='val' or phase='test'")
        source_domain, target_domain = resolve_domains(mode, source_domain, target_domain)
        self.source_domain = source_domain
        self.target_domain = target_domain
        a_paths = list_images(str(Path(dataroot) / phase / source_domain))
        b_paths = list_images(str(Path(dataroot) / phase / target_domain))
        if pairing == 'order':
            if len(a_paths) != len(b_paths):
                raise RuntimeError('Order-based evaluation pairing requires equal source and target counts')
            pairs = list(zip(a_paths, b_paths))
        elif pairing == 'key':
            b_by_key = {}
            for path in b_paths:
                key = _pair_key(path, target_domain)
                if key in b_by_key:
                    raise RuntimeError(f'Duplicate target pairing key: {key}')
                b_by_key[key] = path
            pairs = []
            missing = []
            for path in a_paths:
                key = _pair_key(path, source_domain)
                target = b_by_key.get(key)
                if target is None:
                    missing.append(Path(path).name)
                else:
                    pairs.append((path, target))
            if missing:
                raise RuntimeError(f'Could not establish explicit source-target pairing for {len(missing)} files. First missing files: {missing[:5]}. Use --eval_pairing order only after independently verifying sorted alignment.')
            if len(pairs) != len(b_paths):
                raise RuntimeError('Evaluation source and target sets do not have one-to-one pairing keys')
        else:
            raise ValueError("pairing must be 'key' or 'order'")
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict:
        a_path, b_path = self.pairs[index]
        with Image.open(a_path) as image_a:
            a = self.transform(image_a)
        with Image.open(b_path) as image_b:
            b = self.transform(image_b)
        return {'A': a, 'B': b, 'A_path': a_path, 'B_path': b_path, 'key': _pair_key(a_path, self.source_domain), 'index': index}
