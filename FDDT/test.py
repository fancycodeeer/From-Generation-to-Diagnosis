import argparse
import csv
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image

from data import build_transform
from models import ResnetGenerator, get_norm_layer
from utils_ import psnr_per_image, seed_everything, ssim_per_image, tensor_to_01


IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_images(folder: str) -> List[str]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f'Folder not found: {folder}')

    files = [
        str(x)
        for x in p.iterdir()
        if x.suffix.lower() in IMG_EXTENSIONS
    ]
    files.sort()

    if len(files) == 0:
        raise RuntimeError(f'No image found in: {folder}')

    return files


def get_ab_dirs(
    dataroot: str,
    phase: str,
    mode: str,
    a_dir: str = '',
    b_dir: str = '',
) -> Tuple[str, str]:

    if a_dir and b_dir:
        return (a_dir, b_dir)

    if mode == 't2':
        return (
            os.path.join(dataroot, phase, 't2'),
            os.path.join(dataroot, phase, 'flair'),
        )

    if mode == 't1':
        return (
            os.path.join(dataroot, phase, 't1'),
            os.path.join(dataroot, phase, 't1ce'),
        )

    raise NotImplementedError(
        f'Unsupported mode: {mode}'
    )


def safe_stem(path: str) -> str:
    return Path(str(path)).stem


def pairing_key(path: str) -> str:
    stem = Path(str(path)).stem.lower()

    for domain in (
        't1ce',
        'flair',
        't1',
        't2',
        'ct',
        'mri',
        'mr',
        'pd',
    ):
        stem = re.sub(
            f'(?i)(^|[_-]){re.escape(domain)}(?=$|[_-])',
            '\\1<domain>',
            stem,
        )

    return stem


class PairedABDataset(Dataset):

    def __init__(
        self,
        dataroot: str,
        phase: str,
        mode: str,
        input_nc: int,
        load_size: int,
        crop_size: int,
        a_dir: str = '',
        b_dir: str = '',
    ):
        super().__init__()

        self.dir_A, self.dir_B = get_ab_dirs(
            dataroot=dataroot,
            phase=phase,
            mode=mode,
            a_dir=a_dir,
            b_dir=b_dir,
        )

        A_paths = list_images(self.dir_A)
        B_paths = list_images(self.dir_B)

        b_by_key = {}

        for path in B_paths:
            key = pairing_key(path)

            if key in b_by_key:
                raise RuntimeError(
                    f'Duplicate target pairing key: {key}'
                )

            b_by_key[key] = path

        self.pairs = []
        missing = []

        for path in A_paths:
            target = b_by_key.get(
                pairing_key(path)
            )

            if target is None:
                missing.append(
                    Path(path).name
                )
            else:
                self.pairs.append(
                    (path, target)
                )

        if missing or len(self.pairs) != len(B_paths):
            raise RuntimeError(
                'Could not establish one-to-one evaluation pairing. '
                f'Missing examples: {missing[:5]}'
            )

        self.transform = build_transform(
            phase='val',
            input_nc=input_nc,
            load_size=load_size,
            crop_size=crop_size,
            no_flip=True,
        )

        print('[Dataset] key-based paired test')
        print(
            f'[Dataset] A={len(A_paths)} '
            f'from {self.dir_A}'
        )
        print(
            f'[Dataset] B={len(B_paths)} '
            f'from {self.dir_B}'
        )
        print(
            f'[Dataset] paired={len(self.pairs)}'
        )
        print(
            f'[Dataset] first A: '
            f'{self.pairs[0][0]}'
        )
        print(
            f'[Dataset] first B: '
            f'{self.pairs[0][1]}'
        )
        print(
            f'[Dataset] last  A: '
            f'{self.pairs[-1][0]}'
        )
        print(
            f'[Dataset] last  B: '
            f'{self.pairs[-1][1]}'
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self,
        index: int,
    ) -> Dict:

        A_path, B_path = self.pairs[index]

        A_img = Image.open(A_path)
        B_img = Image.open(B_path)

        A = self.transform(A_img)
        B = self.transform(B_img)

        return {
            'A': A,
            'B': B,
            'A_path': A_path,
            'B_path': B_path,
            'key': safe_stem(A_path),
            'index': index,
        }


def seed_worker(worker_id: int):
    worker_seed = (
        torch.initial_seed()
        % 2 ** 32
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(args) -> DataLoader:

    dataset = PairedABDataset(
        dataroot=args.dataroot,
        phase=args.phase,
        mode=args.mode,
        input_nc=args.input_nc,
        load_size=args.load_size,
        crop_size=args.crop_size,
        a_dir=args.a_dir,
        b_dir=args.b_dir,
    )

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
        worker_init_fn=seed_worker,
    )

    if args.num_workers > 0:
        loader_kwargs[
            'prefetch_factor'
        ] = args.prefetch_factor

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )


def resolve_device(
    device_arg: str,
) -> torch.device:

    if device_arg == 'cuda':
        return torch.device(
            'cuda'
            if torch.cuda.is_available()
            else 'cpu'
        )

    return torch.device(
        device_arg
    )


def looks_like_state_dict(obj) -> bool:
    return (
        isinstance(obj, dict)
        and len(obj) > 0
        and all(
            torch.is_tensor(v)
            for v in obj.values()
        )
    )


def strip_module_prefix(
    state_dict: Dict[
        str,
        torch.Tensor,
    ],
) -> Dict[str, torch.Tensor]:

    out = {}

    for k, v in state_dict.items():
        if k.startswith('module.'):
            out[
                k[len('module.'):]
            ] = v
        else:
            out[k] = v

    return out


def load_generator_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    strict: bool = True,
):

    ckpt = torch.load(
        ckpt_path,
        map_location='cpu',
    )

    if looks_like_state_dict(
        ckpt
    ):
        state = ckpt

    elif isinstance(
        ckpt,
        dict,
    ):
        state = None

        for key in [
            'G_A',
            'netG_A',
            'G',
            'netG',
            'model',
            'state_dict',
        ]:
            if (
                key in ckpt
                and looks_like_state_dict(
                    ckpt[key]
                )
            ):
                state = ckpt[key]
                break

        if state is None:
            raise KeyError(
                'Cannot find generator state_dict '
                f'in {ckpt_path}. '
                f'Available keys: {list(ckpt.keys())}'
            )

    else:
        raise TypeError(
            f'Unsupported checkpoint type: '
            f'{type(ckpt)}'
        )

    state = strip_module_prefix(
        state
    )

    missing, unexpected = (
        model.load_state_dict(
            state,
            strict=strict,
        )
    )

    print(
        f'[Load] {ckpt_path}'
    )

    if not strict:
        print(
            f'[Load] missing={missing}'
        )
        print(
            f'[Load] unexpected={unexpected}'
        )


def build_fid_metric(
    device: torch.device,
    feature: int = 2048,
):

    try:
        from torchmetrics.image.fid import (
            FrechetInceptionDistance,
        )

    except Exception as e:
        raise RuntimeError(
            'FID requires torchmetrics and torch-fidelity. '
            'Install with: '
            'pip install torchmetrics torch-fidelity'
        ) from e

    fid = FrechetInceptionDistance(
        feature=feature,
        normalize=False,
        reset_real_features=True,
    )

    return fid.to(
        device
    )


def to_fid_uint8(
    x_train: torch.Tensor,
    fid_size: int = 299,
) -> torch.Tensor:

    x = tensor_to_01(
        x_train
    )

    if x.size(1) == 1:
        x = x.repeat(
            1,
            3,
            1,
            1,
        )

    elif x.size(1) > 3:
        x = x[:, :3]

    if x.shape[-2:] != (
        fid_size,
        fid_size,
    ):
        x = F.interpolate(
            x.float(),
            size=(
                fid_size,
                fid_size,
            ),
            mode='bilinear',
            align_corners=False,
        )

    return (
        x.mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )


def case_id_from_any_path_or_key(
    x: str,
) -> str:

    stem = Path(
        str(x)
    ).stem

    modality_tokens = {
        't1',
        't1ce',
        't2',
        'flair',
        'seg',
        'mask',
        'ct',
        'mr',
        'mri',
        'pd',
        'adc',
        'dwi',
    }

    us_parts = stem.split('_')

    if us_parts:
        parts = list(
            us_parts
        )

        if (
            parts
            and parts[-1].isdigit()
        ):
            parts = parts[:-1]

        if (
            parts
            and parts[-1].lower()
            in modality_tokens
        ):
            parts = parts[:-1]

        if parts:
            first = parts[0]
            first_hy = first.split('-')

            if (
                first.startswith('BraTS-')
                and len(first_hy) >= 4
            ):
                return '-'.join(
                    first_hy[:4]
                )

            if (
                first.startswith('BraTS')
                and len(parts) >= 2
            ):
                return (
                    f'{parts[0]}_'
                    f'{parts[1]}'
                )

            if first.startswith('IXI'):
                return first.split(
                    '-'
                )[0]

            if len(parts) >= 2:
                return '_'.join(
                    parts[:2]
                )

            if len(parts) == 1:
                return parts[0]

    hy_parts = stem.split('-')

    if hy_parts:
        parts = list(
            hy_parts
        )

        if (
            parts
            and parts[-1].isdigit()
        ):
            parts = parts[:-1]

        if (
            parts
            and parts[-1].lower()
            in modality_tokens
        ):
            parts = parts[:-1]

        if (
            len(parts) >= 4
            and parts[0] == 'BraTS'
        ):
            return '-'.join(
                parts[:4]
            )

        if parts[0].startswith('IXI'):
            return parts[0]

        if len(parts) >= 3:
            return '-'.join(
                parts[:-1]
            )

    return stem


def infer_epoch_from_ckpt(
    ckpt_path: str,
) -> int:

    text = str(
        ckpt_path
    )

    name = Path(
        text
    ).name

    patterns = [
        '(?:^|[_\\-])epoch[_\\-]?(\\d+)(?:[_\\-.]|$)',
        'epoch[_\\-]?(\\d+)',
        'G_epoch[_\\-]?(\\d+)',
    ]

    for target in (
        name,
        text,
    ):
        for pat in patterns:
            m = re.search(
                pat,
                target,
                flags=re.IGNORECASE,
            )

            if m:
                return int(
                    m.group(1)
                )

    return -1


def resolve_lpips_epoch(
    args,
) -> int:

    if (
        getattr(
            args,
            'lpips_epoch',
            -1,
        ) is not None
        and int(
            getattr(
                args,
                'lpips_epoch',
                -1,
            )
        ) >= 0
    ):
        return int(
            args.lpips_epoch
        )

    return infer_epoch_from_ckpt(
        args.ckpt
    )


def build_lpips_metric(
    device: torch.device,
    net: str = 'alex',
):

    try:
        import lpips

    except Exception as e:
        raise RuntimeError(
            'LPIPS requires lpips. '
            'Install with: pip install lpips'
        ) from e

    metric = lpips.LPIPS(
        net=net
    ).to(
        device
    )

    metric.eval()

    for p in metric.parameters():
        p.requires_grad_(
            False
        )

    return metric


def to_lpips_input(
    x_train: torch.Tensor,
) -> torch.Tensor:

    x = (
        x_train
        .detach()
        .float()
        .clamp(
            -1.0,
            1.0,
        )
    )

    if x.size(1) == 1:
        x = x.repeat(
            1,
            3,
            1,
            1,
        )

    elif x.size(1) > 3:
        x = x[:, :3]

    return x


@torch.no_grad()
def lpips_per_image(
    lpips_metric,
    fake_B: torch.Tensor,
    real_B: torch.Tensor,
) -> torch.Tensor:

    fake_lpips = to_lpips_input(
        fake_B
    )

    real_lpips = to_lpips_input(
        real_B
    )

    vals = lpips_metric(
        fake_lpips,
        real_lpips,
    )

    return (
        vals
        .view(
            vals.size(0)
        )
        .detach()
        .float()
    )


def compute_case_metrics_from_slice_rows(
    slice_rows: List[Dict],
) -> Tuple[List[Dict], Dict]:

    grouped = defaultdict(
        list
    )

    for row in slice_rows:
        grouped[
            row['case_id']
        ].append(
            row
        )

    case_rows = []

    for case_id in sorted(
        grouped.keys()
    ):
        rows = grouped[
            case_id
        ]

        ssim_arr = np.asarray(
            [
                float(x['ssim'])
                for x in rows
            ],
            dtype=np.float64,
        )

        psnr_arr = np.asarray(
            [
                float(x['psnr'])
                for x in rows
            ],
            dtype=np.float64,
        )

        lpips_arr = np.asarray(
            [
                float(x['lpips'])
                for x in rows
            ],
            dtype=np.float64,
        )

        case_rows.append(
            {
                'case_id':
                    case_id,

                'n_slices':
                    int(len(rows)),

                'ssim_mean':
                    float(
                        ssim_arr.mean()
                    ),

                'ssim_std_slice':
                    float(
                        ssim_arr.std(
                            ddof=1
                        )
                    )
                    if len(ssim_arr) > 1
                    else 0.0,

                'psnr_mean':
                    float(
                        psnr_arr.mean()
                    ),

                'psnr_std_slice':
                    float(
                        psnr_arr.std(
                            ddof=1
                        )
                    )
                    if len(psnr_arr) > 1
                    else 0.0,

                'lpips_mean':
                    float(
                        lpips_arr.mean()
                    ),

                'lpips_std_slice':
                    float(
                        lpips_arr.std(
                            ddof=1
                        )
                    )
                    if len(lpips_arr) > 1
                    else 0.0,
            }
        )

    case_ssim = np.asarray(
        [
            x['ssim_mean']
            for x in case_rows
        ],
        dtype=np.float64,
    )

    case_psnr = np.asarray(
        [
            x['psnr_mean']
            for x in case_rows
        ],
        dtype=np.float64,
    )

    case_lpips = np.asarray(
        [
            x['lpips_mean']
            for x in case_rows
        ],
        dtype=np.float64,
    )

    summary = {
        'n_cases':
            int(
                len(case_rows)
            ),

        'case_ssim_mean':
            float(
                case_ssim.mean()
            )
            if len(case_ssim)
            else float('nan'),

        'case_ssim_std':
            float(
                case_ssim.std(
                    ddof=1
                )
            )
            if len(case_ssim) > 1
            else 0.0,

        'case_psnr_mean':
            float(
                case_psnr.mean()
            )
            if len(case_psnr)
            else float('nan'),

        'case_psnr_std':
            float(
                case_psnr.std(
                    ddof=1
                )
            )
            if len(case_psnr) > 1
            else 0.0,

        'case_lpips_mean':
            float(
                case_lpips.mean()
            )
            if len(case_lpips)
            else float('nan'),

        'case_lpips_std':
            float(
                case_lpips.std(
                    ddof=1
                )
            )
            if len(case_lpips) > 1
            else 0.0,
    }

    return (
        case_rows,
        summary,
    )


def load_ref_case_lpips_csv(
    path: str,
) -> Dict[str, float]:

    ref = {}

    with open(
        path,
        'r',
        newline='',
    ) as f:

        reader = csv.DictReader(
            f
        )

        if (
            'case_id'
            not in reader.fieldnames
            or 'lpips_mean'
            not in reader.fieldnames
        ):
            raise ValueError(
                'Reference case CSV must contain columns: '
                'case_id, lpips_mean. '
                f'Got: {reader.fieldnames}'
            )

        for row in reader:
            ref[
                str(
                    row['case_id']
                )
            ] = float(
                row['lpips_mean']
            )

    return ref


def compute_lpips_paired_ttest_ci(
    case_rows: List[Dict],
    ref_case_csv: str,
    alpha: float = 0.05,
) -> Dict:

    try:
        from scipy import stats

    except Exception as e:
        raise RuntimeError(
            'LPIPS paired t-test and 95% CI require scipy. '
            'Install with: pip install scipy'
        ) from e

    ref = load_ref_case_lpips_csv(
        ref_case_csv
    )

    diffs = []

    for row in case_rows:
        case_id = str(
            row['case_id']
        )

        if case_id in ref:
            diffs.append(
                float(
                    row['lpips_mean']
                )
                -
                float(
                    ref[case_id]
                )
            )

    diffs = np.asarray(
        diffs,
        dtype=np.float64,
    )

    diffs = diffs[
        np.isfinite(
            diffs
        )
    ]

    n = int(
        len(diffs)
    )

    if n < 2:
        return {
            'lpips_paired_ref_case_csv':
                ref_case_csv,

            'lpips_paired_n_cases':
                n,

            'lpips_paired_mean_diff_current_minus_ref':
                float(
                    diffs.mean()
                )
                if n == 1
                else float('nan'),

            'lpips_paired_std_diff':
                0.0
                if n == 1
                else float('nan'),

            'lpips_paired_sem_diff':
                float('nan'),

            'lpips_paired_t_stat':
                float('nan'),

            'lpips_paired_p_value':
                float('nan'),

            'lpips_paired_ci95_low':
                float('nan'),

            'lpips_paired_ci95_high':
                float('nan'),
        }

    mean_diff = float(
        diffs.mean()
    )

    std_diff = float(
        diffs.std(
            ddof=1
        )
    )

    sem_diff = float(
        std_diff
        / np.sqrt(n)
    )

    df = n - 1

    t_stat, p_value = (
        stats.ttest_1samp(
            diffs,
            popmean=0.0,
        )
    )

    t_crit = float(
        stats.t.ppf(
            1.0
            -
            alpha / 2.0,
            df,
        )
    )

    ci_low = (
        mean_diff
        -
        t_crit
        *
        sem_diff
    )

    ci_high = (
        mean_diff
        +
        t_crit
        *
        sem_diff
    )

    return {
        'lpips_paired_ref_case_csv':
            ref_case_csv,

        'lpips_paired_n_cases':
            n,

        'lpips_paired_mean_diff_current_minus_ref':
            mean_diff,

        'lpips_paired_std_diff':
            std_diff,

        'lpips_paired_sem_diff':
            sem_diff,

        'lpips_paired_t_stat':
            float(
                t_stat
            ),

        'lpips_paired_p_value':
            float(
                p_value
            ),

        'lpips_paired_ci95_low':
            float(
                ci_low
            ),

        'lpips_paired_ci95_high':
            float(
                ci_high
            ),
    }


def save_tensor_png(
    x01: torch.Tensor,
    path: str,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    x01 = (
        x01
        .detach()
        .float()
        .cpu()
        .clamp(
            0.0,
            1.0,
        )
    )

    if x01.dim() != 3:
        raise ValueError(
            'Expected CHW tensor, '
            f'got {tuple(x01.shape)}'
        )

    img = to_pil_image(
        x01
    )

    img.save(
        str(path)
    )


def append_csv(
    path: str,
    row: Dict,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not path.exists()
    )

    with path.open(
        'a',
        newline='',
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                row.keys()
            ),
        )

        if write_header:
            writer.writeheader()

        writer.writerow(
            row
        )


def save_slice_csv(
    path: str,
    rows: List[Dict],
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        'w',
        newline='',
    ) as f:

        fieldnames = [
            'index',
            'case_id',
            'key',
            'A_path',
            'B_path',
            'ssim',
            'psnr',
            'lpips',
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def save_case_csv(
    path: str,
    case_rows: List[Dict],
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        'case_id',
        'n_slices',
        'ssim_mean',
        'ssim_std_slice',
        'psnr_mean',
        'psnr_std_slice',
        'lpips_mean',
        'lpips_std_slice',
    ]

    with path.open(
        'w',
        newline='',
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in case_rows:
            writer.writerow(
                row
            )


def make_lpips_only_case_rows(
    case_rows: List[Dict],
    epoch: int,
) -> List[Dict]:

    return [
        {
            'epoch':
                int(epoch),

            'case_id':
                str(
                    row['case_id']
                ),

            'n_slices':
                int(
                    row['n_slices']
                ),

            'lpips_mean':
                float(
                    row['lpips_mean']
                ),

            'lpips_std_slice':
                float(
                    row[
                        'lpips_std_slice'
                    ]
                ),
        }
        for row in case_rows
    ]


def upsert_csv_rows_by_epoch(
    path: str,
    rows: List[Dict],
    fieldnames: List[str],
    epoch: int,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    kept_rows = []

    if path.exists():

        with path.open(
            'r',
            newline='',
        ) as f:

            reader = csv.DictReader(
                f
            )

            for row in reader:

                if str(
                    row.get(
                        'epoch',
                        '',
                    )
                ) != str(epoch):

                    kept_rows.append(
                        {
                            k: row.get(
                                k,
                                '',
                            )
                            for k
                            in fieldnames
                        }
                    )

    new_rows = [
        {
            k: row.get(
                k,
                '',
            )
            for k
            in fieldnames
        }
        for row in rows
    ]

    def sort_key(row):
        try:
            ep = int(
                row.get(
                    'epoch',
                    -1,
                )
            )

        except Exception:
            ep = -1

        return (
            ep,
            str(
                row.get(
                    'case_id',
                    '',
                )
            ),
        )

    all_rows = sorted(
        kept_rows
        +
        new_rows,
        key=sort_key,
    )

    with path.open(
        'w',
        newline='',
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )


def save_lpips_only_outputs(
    args,
    case_rows: List[Dict],
    case_summary: Dict,
    lpips_mean_slice: float,
    lpips_std_slice: float,
    n_images: int,
) -> int:

    epoch = resolve_lpips_epoch(
        args
    )

    if epoch < 0:
        print(
            '[LPIPS] Skip separate LPIPS CSV saving '
            'because epoch cannot be resolved. '
            'Pass --lpips_epoch N when testing '
            'best/latest checkpoints.'
        )

        return epoch

    lpips_only_case_rows = (
        make_lpips_only_case_rows(
            case_rows,
            epoch=epoch,
        )
    )

    if args.lpips_case_csv:

        upsert_csv_rows_by_epoch(
            path=args.lpips_case_csv,
            rows=lpips_only_case_rows,
            fieldnames=[
                'epoch',
                'case_id',
                'n_slices',
                'lpips_mean',
                'lpips_std_slice',
            ],
            epoch=epoch,
        )

        print(
            '[LPIPS] Saved separate '
            'case-level LPIPS CSV: '
            f'{args.lpips_case_csv}'
        )

    if args.lpips_summary_csv:

        summary_row = {
            'epoch':
                int(epoch),

            'model':
                'cyclegan',

            'ckpt':
                args.ckpt,

            'mode':
                args.mode,

            'phase':
                args.phase,

            'n_images':
                int(n_images),

            'n_cases':
                int(
                    case_summary[
                        'n_cases'
                    ]
                ),

            'lpips_mean_slice':
                float(
                    lpips_mean_slice
                ),

            'lpips_std_slice':
                float(
                    lpips_std_slice
                ),

            'lpips_mean_case':
                float(
                    case_summary[
                        'case_lpips_mean'
                    ]
                ),

            'lpips_std_case':
                float(
                    case_summary[
                        'case_lpips_std'
                    ]
                ),

            'lpips_net':
                args.lpips_net,
        }

        upsert_csv_rows_by_epoch(
            path=args.lpips_summary_csv,
            rows=[
                summary_row
            ],
            fieldnames=[
                'epoch',
                'model',
                'ckpt',
                'mode',
                'phase',
                'n_images',
                'n_cases',
                'lpips_mean_slice',
                'lpips_std_slice',
                'lpips_mean_case',
                'lpips_std_case',
                'lpips_net',
            ],
            epoch=epoch,
        )

        print(
            '[LPIPS] Saved separate '
            'summary LPIPS CSV: '
            f'{args.lpips_summary_csv}'
        )

    return epoch


@torch.inference_mode()
def evaluate(
    netG,
    loader,
    device,
    args,
):

    netG.eval()

    fid = build_fid_metric(
        device,
        feature=args.fid_feature,
    )

    lpips_metric = build_lpips_metric(
        device,
        net=args.lpips_net,
    )

    all_ssim = []
    all_psnr = []
    all_lpips = []
    slice_rows = []

    if args.save_images:
        Path(
            args.image_dir
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    n_images = 0

    for batch_idx, batch in enumerate(
        loader
    ):

        if (
            args.max_batches > 0
            and batch_idx
            >= args.max_batches
        ):
            break

        real_A = batch[
            'A'
        ].to(
            device,
            non_blocking=True,
        )

        real_B = batch[
            'B'
        ].to(
            device,
            non_blocking=True,
        )

        if args.channels_last:

            real_A = (
                real_A
                .contiguous(
                    memory_format=
                    torch.channels_last
                )
            )

            real_B = (
                real_B
                .contiguous(
                    memory_format=
                    torch.channels_last
                )
            )

        with autocast(
            enabled=(
                args.amp
                and device.type == 'cuda'
            )
        ):
            fake_B = netG(
                real_A
            )

        fake_B = (
            fake_B
            .detach()
            .clamp(
                -1.0,
                1.0,
            )
        )

        fake01 = tensor_to_01(
            fake_B
        )

        realB01 = tensor_to_01(
            real_B
        )

        realA01 = tensor_to_01(
            real_A
        )

        ssim_vals = (
            ssim_per_image(
                fake01,
                realB01,
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        psnr_vals = (
            psnr_per_image(
                fake01,
                realB01,
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        lpips_vals = (
            lpips_per_image(
                lpips_metric,
                fake_B,
                real_B,
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        all_ssim.extend(
            [
                float(x)
                for x in ssim_vals
            ]
        )

        all_psnr.extend(
            [
                float(x)
                for x in psnr_vals
            ]
        )

        all_lpips.extend(
            [
                float(x)
                for x in lpips_vals
            ]
        )

        fid.update(
            to_fid_uint8(
                real_B,
                args.fid_size,
            ),
            real=True,
        )

        fid.update(
            to_fid_uint8(
                fake_B,
                args.fid_size,
            ),
            real=False,
        )

        keys = batch.get(
            'key',
            [
                str(i)
                for i in range(
                    real_A.size(0)
                )
            ],
        )

        A_paths = batch.get(
            'A_path',
            ['']
            *
            real_A.size(0),
        )

        B_paths = batch.get(
            'B_path',
            ['']
            *
            real_A.size(0),
        )

        for i in range(
            real_A.size(0)
        ):

            key = str(
                keys[i]
            )

            case_id = (
                case_id_from_any_path_or_key(
                    key
                )
            )

            slice_rows.append(
                {
                    'index':
                        int(n_images),

                    'case_id':
                        case_id,

                    'key':
                        key,

                    'A_path':
                        str(
                            A_paths[i]
                        ),

                    'B_path':
                        str(
                            B_paths[i]
                        ),

                    'ssim':
                        float(
                            ssim_vals[i]
                        ),

                    'psnr':
                        float(
                            psnr_vals[i]
                        ),

                    'lpips':
                        float(
                            lpips_vals[i]
                        ),
                }
            )

            if args.save_images:

                save_tensor_png(
                    fake01[i],
                    os.path.join(
                        args.image_dir,
                        'fake_B',
                        f'{key}.png',
                    ),
                )

                if args.save_real:

                    save_tensor_png(
                        realB01[i],
                        os.path.join(
                            args.image_dir,
                            'real_B',
                            f'{key}.png',
                        ),
                    )

                if args.save_input:

                    save_tensor_png(
                        realA01[i],
                        os.path.join(
                            args.image_dir,
                            'real_A',
                            f'{key}.png',
                        ),
                    )

            n_images += 1

        if (
            args.print_freq > 0
            and (
                batch_idx + 1
            )
            % args.print_freq
            == 0
        ):
            print(
                f'[Test] batch={batch_idx + 1}, '
                f'images={n_images}'
            )

    fid_value = float(
        fid.compute()
        .detach()
        .cpu()
        .item()
    )

    ssim_mean = (
        float(
            np.mean(
                all_ssim
            )
        )
        if len(all_ssim)
        else 0.0
    )

    ssim_std = (
        float(
            np.std(
                all_ssim,
                ddof=1,
            )
        )
        if len(all_ssim) > 1
        else 0.0
    )

    psnr_mean = (
        float(
            np.mean(
                all_psnr
            )
        )
        if len(all_psnr)
        else 0.0
    )

    psnr_std = (
        float(
            np.std(
                all_psnr,
                ddof=1,
            )
        )
        if len(all_psnr) > 1
        else 0.0
    )

    lpips_mean_slice = (
        float(
            np.mean(
                all_lpips
            )
        )
        if len(all_lpips)
        else 0.0
    )

    lpips_std_slice = (
        float(
            np.std(
                all_lpips,
                ddof=1,
            )
        )
        if len(all_lpips) > 1
        else 0.0
    )

    case_rows, case_summary = (
        compute_case_metrics_from_slice_rows(
            slice_rows
        )
    )

    paired_ttest = {}

    if args.lpips_ref_case_csv:

        paired_ttest = (
            compute_lpips_paired_ttest_ci(
                case_rows=case_rows,
                ref_case_csv=
                    args.lpips_ref_case_csv,
                alpha=0.05,
            )
        )

    lpips_epoch = (
        save_lpips_only_outputs(
            args=args,
            case_rows=case_rows,
            case_summary=case_summary,
            lpips_mean_slice=
                lpips_mean_slice,
            lpips_std_slice=
                lpips_std_slice,
            n_images=n_images,
        )
    )

    summary = {
        'model':
            'cyclegan',

        'ckpt':
            args.ckpt,

        'mode':
            args.mode,

        'phase':
            args.phase,

        'n_images':
            int(n_images),

        'fid':
            fid_value,

        'ssim_mean_slice':
            ssim_mean,

        'ssim_std_slice':
            ssim_std,

        'psnr_mean_slice':
            psnr_mean,

        'psnr_std_slice':
            psnr_std,

        'lpips_mean_slice':
            lpips_mean_slice,

        'lpips_std_slice':
            lpips_std_slice,

        'n_cases':
            case_summary[
                'n_cases'
            ],

        'case_ssim_mean':
            case_summary[
                'case_ssim_mean'
            ],

        'case_ssim_std':
            case_summary[
                'case_ssim_std'
            ],

        'case_psnr_mean':
            case_summary[
                'case_psnr_mean'
            ],

        'case_psnr_std':
            case_summary[
                'case_psnr_std'
            ],

        'case_lpips_mean':
            case_summary[
                'case_lpips_mean'
            ],

        'case_lpips_std':
            case_summary[
                'case_lpips_std'
            ],

        **paired_ttest,

        'fid_size':
            int(
                args.fid_size
            ),

        'load_size':
            int(
                args.load_size
            ),

        'crop_size':
            int(
                args.crop_size
            ),

        'lpips_net':
            args.lpips_net,
    }

    print(
        '=' * 80
    )

    print(
        f'[CycleGAN] n_images={n_images}'
    )

    print(
        f"[CycleGAN] n_cases="
        f"{case_summary['n_cases']}"
    )

    print(
        f'[CycleGAN] FID='
        f'{fid_value:.6f}'
    )

    print(
        f'[CycleGAN] SSIM slice='
        f'{ssim_mean:.6f} '
        f'± {ssim_std:.6f}'
    )

    print(
        f'[CycleGAN] PSNR slice='
        f'{psnr_mean:.6f} '
        f'± {psnr_std:.6f}'
    )

    print(
        f'[CycleGAN] LPIPS slice='
        f'{lpips_mean_slice:.6f} '
        f'± {lpips_std_slice:.6f}'
    )

    print(
        f"[CycleGAN] LPIPS case="
        f"{case_summary['case_lpips_mean']:.6f} "
        f"± "
        f"{case_summary['case_lpips_std']:.6f}"
    )

    if lpips_epoch >= 0:

        print(
            '[CycleGAN] separate LPIPS CSV '
            f'epoch={lpips_epoch}'
        )

    if paired_ttest:

        print(
            '[CycleGAN] LPIPS paired '
            'two-sided t-test current-ref: '
            f"n="
            f"{paired_ttest['lpips_paired_n_cases']}, "
            f"mean_diff="
            f"{paired_ttest['lpips_paired_mean_diff_current_minus_ref']:.6f}, "
            f"p="
            f"{paired_ttest['lpips_paired_p_value']:.6g}, "
            f"95%CI=["
            f"{paired_ttest['lpips_paired_ci95_low']:.6f}, "
            f"{paired_ttest['lpips_paired_ci95_high']:.6f}]"
        )

    print(
        '=' * 80
    )

    if args.metrics_csv:
        append_csv(
            args.metrics_csv,
            summary,
        )

    if args.slice_csv:
        save_slice_csv(
            args.slice_csv,
            slice_rows,
        )

    if args.case_csv:
        save_case_csv(
            args.case_csv,
            case_rows,
        )

    return summary


def parse_args():

    parser = argparse.ArgumentParser(
        'Test CycleGAN/FDIT A->B with FID'
    )

    parser.add_argument(
        '--dataroot',
        type=str,
        default='',
    )

    parser.add_argument(
        '--phase',
        type=str,
        default='val',
        choices=[
            'val',
            'test',
        ],
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='t2',
        choices=[
            't1',
            't2',
        ],
    )

    parser.add_argument(
        '--a_dir',
        type=str,
        default='',
        help='Optional A folder override.',
    )

    parser.add_argument(
        '--b_dir',
        type=str,
        default='',
        help='Optional B folder override.',
    )

    parser.add_argument(
        '--ckpt',
        type=str,
        default='',
        help='epoch/best/latest G weight file.',
    )

    parser.add_argument(
        '--lpips_case_csv',
        type=str,
        default='',
    )

    parser.add_argument(
        '--lpips_summary_csv',
        type=str,
        default='',
    )

    parser.add_argument(
        '--lpips_ref_case_csv',
        type=str,
        default='',
        help=(
            'Optional reference case-level CSV '
            'for true paired t-test. '
            'If provided, script tests '
            'current_case_lpips - ref_case_lpips.'
        ),
    )

    parser.add_argument(
        '--input_nc',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--output_nc',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--ngf',
        type=int,
        default=64,
    )

    parser.add_argument(
        '--netG_blocks',
        type=int,
        default=9,
    )

    parser.add_argument(
        '--norm',
        type=str,
        default='instance',
        choices=[
            'instance',
            'batch',
            'none',
        ],
    )

    parser.add_argument(
        '--load_size',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--crop_size',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
    )

    parser.add_argument(
        '--prefetch_factor',
        type=int,
        default=4,
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    parser.add_argument(
        '--amp',
        action='store_true',
    )

    parser.add_argument(
        '--channels_last',
        action='store_true',
    )

    parser.add_argument(
        '--fid_size',
        type=int,
        default=299,
    )

    parser.add_argument(
        '--fid_feature',
        type=int,
        default=2048,
    )

    parser.add_argument(
        '--lpips_net',
        type=str,
        default='alex',
        choices=[
            'alex',
            'vgg',
            'squeeze',
        ],
    )

    parser.add_argument(
        '--lpips_epoch',
        type=int,
        default=-1,
        help=(
            'Epoch id written to separate LPIPS CSV. '
            'If -1, infer from --ckpt filename. '
            'Required for best/latest ckpts.'
        ),
    )

    parser.add_argument(
        '--save_images',
        action='store_true',
    )

    parser.add_argument(
        '--save_real',
        action='store_true',
    )

    parser.add_argument(
        '--save_input',
        action='store_true',
    )

    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
    )

    parser.add_argument(
        '--metrics_csv',
        type=str,
        default='',
    )

    parser.add_argument(
        '--slice_csv',
        type=str,
        default='',
    )

    parser.add_argument(
        '--case_csv',
        type=str,
        default='',
    )

    parser.add_argument(
        '--max_batches',
        type=int,
        default=0,
        help='0 means full test set.',
    )

    parser.add_argument(
        '--print_freq',
        type=int,
        default=20,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    seed_everything(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    torch.backends.cudnn.benchmark = True

    try:
        torch.set_float32_matmul_precision(
            'high'
        )

    except Exception:
        pass

    loader = make_loader(
        args
    )

    norm_layer = get_norm_layer(
        args.norm
    )

    netG = ResnetGenerator(
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        ngf=args.ngf,
        norm_layer=norm_layer,
        use_dropout=False,
        n_blocks=args.netG_blocks,
    ).to(
        device
    )

    load_generator_weights(
        netG,
        args.ckpt,
        strict=True,
    )

    if args.channels_last:

        netG = netG.to(
            memory_format=
            torch.channels_last
        )

    evaluate(
        netG,
        loader,
        device,
        args,
    )


if __name__ == '__main__':
    main()