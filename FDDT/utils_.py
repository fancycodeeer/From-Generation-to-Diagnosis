import csv
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as skimage_ssim

class ImagePool:

    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images: List[torch.Tensor] = []

    @torch.no_grad()
    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return images.detach()
        return_images = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image.clone())
                return_images.append(image)
            elif random.random() > 0.5:
                random_id = random.randint(0, self.pool_size - 1)
                tmp = self.images[random_id].clone()
                self.images[random_id] = image.clone()
                return_images.append(tmp)
            else:
                return_images.append(image)
        return torch.cat(return_images, dim=0)

class GANLoss(nn.Module):

    def __init__(self, gan_mode: str='lsgan'):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f'Unsupported gan_mode: {gan_mode}')

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            return torch.ones_like(prediction)
        return torch.zeros_like(prediction)

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

def set_requires_grad(nets: Iterable[nn.Module], requires_grad: bool=False) -> None:
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, 'module') else model

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_main_process() -> bool:
    return not is_dist_avail_and_initialized() or dist.get_rank() == 0

def reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def optimizer_to(optim: torch.optim.Optimizer, device: torch.device) -> None:
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def get_linear_decay_lr(epoch: int, base_lr: float, n_epochs: int, n_epochs_decay: int) -> float:
    if epoch <= n_epochs:
        return base_lr
    if n_epochs_decay <= 0:
        return base_lr
    progress = min(max(epoch - n_epochs, 0), n_epochs_decay)
    return base_lr * (1.0 - progress / float(n_epochs_decay))

def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = lr

def tensor_to_01(x: torch.Tensor) -> torch.Tensor:
    return (x.detach().float() + 1.0).mul(0.5).clamp(0.0, 1.0)

def gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)
    return w.expand(channels, 1, window_size, window_size).contiguous()

@torch.no_grad()
def ssim_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float=1.0) -> torch.Tensor:
    device = x.device
    x_np = x.detach().float().clamp(0.0, 1.0).cpu().numpy()
    y_np = y.detach().float().clamp(0.0, 1.0).cpu().numpy()
    b, c, h, w = x_np.shape
    min_hw = min(h, w)
    win_size = min(7, min_hw)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        raise ValueError(f'Image is too small for SSIM: H={h}, W={w}. Need at least 3x3.')
    vals = []
    for i in range(b):
        if c == 1:
            xi = x_np[i, 0]
            yi = y_np[i, 0]
            score = skimage_ssim(yi, xi, data_range=data_range, win_size=win_size, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        else:
            channel_scores = []
            for ch in range(c):
                xi = x_np[i, ch]
                yi = y_np[i, ch]
                ch_score = skimage_ssim(yi, xi, data_range=data_range, win_size=win_size, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
                channel_scores.append(ch_score)
            score = float(np.mean(channel_scores))
        vals.append(float(score))
    return torch.tensor(vals, dtype=torch.float32, device=device)

@torch.no_grad()
def psnr_per_image(x: torch.Tensor, y: torch.Tensor, data_range: float=1.0) -> torch.Tensor:
    x = x.float()
    y = y.float()
    mse = F.mse_loss(x, y, reduction='none').flatten(1).mean(dim=1)
    return 10.0 * torch.log10(data_range ** 2 / (mse + 1e-12))

def append_csv(path: str, row: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def save_checkpoint(path: str, state: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def train_tensor_to_255(x: torch.Tensor, clamp: bool=False) -> torch.Tensor:
    x = x.float()
    if clamp:
        x = x.clamp(-1.0, 1.0)
    return (x + 1.0) * 0.5 * 255.0

def image_255_to_train_tensor(x_255: torch.Tensor, clamp: bool=False) -> torch.Tensor:
    if clamp:
        x_255 = x_255.clamp(0.0, 255.0)
    return x_255 / 127.5 - 1.0

def make_gaussian_freq_masks_fft(size, D, device, dtype=torch.float32):
    B, C, H, W = size
    yy = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)
    cy = H / 2.0
    cx = W / 2.0
    dist2 = (yy - cy).pow(2) + (xx - cx).pow(2)
    D = torch.as_tensor(float(D), device=device, dtype=dtype).clamp_min(1e-06)
    mask_L = torch.exp(-dist2 / (2.0 * D.pow(2)))
    mask_H = 1.0 - mask_L
    mask_L = mask_L.expand(B, C, H, W).contiguous()
    mask_H = mask_H.expand(B, C, H, W).contiguous()
    return (mask_H, mask_L)

def frequency_decompose_train_space(x: torch.Tensor, D: float=30.0, clamp_input: bool=False, clamp_component: bool=False):
    x_255 = train_tensor_to_255(x, clamp=clamp_input)
    B, C, H, W = x_255.shape
    x_255 = x_255.float()
    freq = torch.fft.fft2(x_255)
    freq = torch.fft.fftshift(freq, dim=(-2, -1))
    mask_H, mask_L = make_gaussian_freq_masks_fft(size=freq.shape, D=D, device=x.device, dtype=x_255.dtype)
    freq_H = freq * mask_H
    freq_L = freq * mask_L
    freq_H = torch.fft.ifftshift(freq_H, dim=(-2, -1))
    freq_L = torch.fft.ifftshift(freq_L, dim=(-2, -1))
    high_255 = torch.abs(torch.fft.ifft2(freq_H).real)
    low_255 = torch.abs(torch.fft.ifft2(freq_L).real)
    high_train = image_255_to_train_tensor(high_255, clamp=clamp_component)
    low_train = image_255_to_train_tensor(low_255, clamp=clamp_component)
    return (high_train, low_train, high_255, low_255)

def fdit_loss_from_train_space(fake: torch.Tensor, source: torch.Tensor, D: float=30.0, lambda_high: float=1.0, lambda_low: float=0.0, detach_source: bool=True):
    if detach_source:
        source = source.detach()
    fake_H, fake_L, _, _ = frequency_decompose_train_space(fake, D=D, clamp_input=False, clamp_component=False)
    src_H, src_L, _, _ = frequency_decompose_train_space(source, D=D, clamp_input=False, clamp_component=False)
    loss_high = F.l1_loss(fake_H, src_H)
    loss_low = F.l1_loss(fake_L, src_L)
    loss = lambda_high * loss_high + lambda_low * loss_low
    return (loss, loss_high, loss_low)
