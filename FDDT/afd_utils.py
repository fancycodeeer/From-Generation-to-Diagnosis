from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int=2):
        super().__init__()
        groups = min(8, out_channels)
        while groups > 1 and out_channels % groups != 0:
            groups -= 1
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(groups, out_channels), nn.SiLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class AdaptiveFrequencyDecomposition(nn.Module):

    def __init__(self, input_nc: int, embed_dim: int, num_heads: int, high_min: float, high_max: float, low_min: float, low_max: float):
        super().__init__()
        if input_nc < 1:
            raise ValueError('input_nc must be positive')
        if embed_dim < 8 or embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be at least 8 and divisible by num_heads')
        if not high_min < high_max:
            raise ValueError('high_min must be smaller than high_max')
        if not low_min < low_max:
            raise ValueError('low_min must be smaller than low_max')
        self.input_nc = int(input_nc)
        self.high_min = float(high_min)
        self.high_max = float(high_max)
        self.low_min = float(low_min)
        self.low_max = float(low_max)
        c1 = max(embed_dim // 4, 8)
        c2 = max(embed_dim // 2, 8)
        self.spatial_encoder = nn.Sequential(ConvBlock(1, c1), ConvBlock(c1, c2), ConvBlock(c2, embed_dim))
        self.frequency_encoder = nn.Sequential(ConvBlock(2, c1), ConvBlock(c1, c2), ConvBlock(c2, embed_dim))
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        fuse_groups = min(8, embed_dim)
        while fuse_groups > 1 and embed_dim % fuse_groups != 0:
            fuse_groups -= 1
        self.fuse = nn.Sequential(nn.Conv2d(embed_dim * 2, embed_dim, 3, padding=1, bias=False), nn.GroupNorm(fuse_groups, embed_dim), nn.SiLU(inplace=True))
        self.high_head = nn.LazyLinear(1)
        self.low_head = nn.LazyLinear(1)

    @staticmethod
    def _grayscale(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'Expected [B,C,H,W], got {tuple(x.shape)}')
        if x.shape[1] == 1:
            return x
        return x.mean(dim=1, keepdim=True)

    @staticmethod
    def _frequency_features(x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fftshift(torch.fft.fft2(x.float()), dim=(-2, -1))
        magnitude = torch.log(torch.abs(spectrum) + 1.0)
        phase = torch.angle(spectrum)
        return torch.cat([magnitude, phase], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_gray = self._grayscale(x).float()
        spatial = self.spatial_encoder(x_gray)
        frequency = self.frequency_encoder(self._frequency_features(x_gray))
        q = spatial.flatten(2).transpose(1, 2)
        kv = frequency.flatten(2).transpose(1, 2)
        attention, _ = self.cross_attention(q, kv, kv, need_weights=False)
        attention = attention.transpose(1, 2).reshape_as(spatial)
        fused = self.fuse(torch.cat([spatial, attention], dim=1))
        vector = fused.flatten(1)
        high_score = torch.sigmoid(self.high_head(vector)).squeeze(1)
        low_score = torch.sigmoid(self.low_head(vector)).squeeze(1)
        high = self.high_min + (self.high_max - self.high_min) * high_score
        low = self.low_min + (self.low_max - self.low_min) * low_score
        return (high, low)

def make_gaussian_frequency_masks(shape: torch.Size, low_cutoff: torch.Tensor, high_cutoff: torch.Tensor, device: torch.device, dtype: torch.dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(shape) != 4:
        raise ValueError('shape must be [B,C,H,W]')
    batch, channels, height, width = shape
    yy = torch.arange(height, device=device, dtype=dtype).view(1, 1, height, 1)
    xx = torch.arange(width, device=device, dtype=dtype).view(1, 1, 1, width)
    cy = height // 2
    cx = width // 2
    radius2 = (yy - cy).pow(2) + (xx - cx).pow(2)
    low = torch.as_tensor(low_cutoff, device=device, dtype=dtype).reshape(-1, 1, 1, 1)
    high = torch.as_tensor(high_cutoff, device=device, dtype=dtype).reshape(-1, 1, 1, 1)
    if low.numel() == 1:
        low = low.expand(batch, 1, 1, 1)
    if high.numel() == 1:
        high = high.expand(batch, 1, 1, 1)
    if low.shape[0] != batch or high.shape[0] != batch:
        raise ValueError('Cutoff batch size must be one or match the image batch size')
    low = low.clamp_min(torch.finfo(dtype).eps)
    high = high.clamp_min(torch.finfo(dtype).eps)
    low_mask = torch.exp(-radius2 / (2.0 * low.pow(2)))
    high_mask = 1.0 - torch.exp(-radius2 / (2.0 * high.pow(2)))
    return (low_mask.expand(batch, channels, height, width), high_mask.expand(batch, channels, height, width))

def frequency_decompose(x: torch.Tensor, low_cutoff: torch.Tensor, high_cutoff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    original_dtype = x.dtype
    x32 = x.float()
    spectrum = torch.fft.fftshift(torch.fft.fft2(x32), dim=(-2, -1))
    low_mask, high_mask = make_gaussian_frequency_masks(spectrum.shape, low_cutoff, high_cutoff, x.device, spectrum.real.dtype)
    low = torch.fft.ifft2(torch.fft.ifftshift(spectrum * low_mask, dim=(-2, -1))).real
    high = torch.fft.ifft2(torch.fft.ifftshift(spectrum * high_mask, dim=(-2, -1))).real
    return (low.to(original_dtype), high.to(original_dtype))

def adaptive_frequency_consistency_loss(generator: nn.Module, afd: AdaptiveFrequencyDecomposition, source: torch.Tensor, full_translation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_high, source_low = afd(source)
    source_high_detached = source_high.detach()
    source_low_detached = source_low.detach()
    source_low_band, source_high_band = frequency_decompose(source, source_low_detached, source_high_detached)
    target_high, target_low = afd(full_translation)
    target_low_band, target_high_band = frequency_decompose(full_translation, target_low, target_high)
    translated_low = generator(source_low_band)
    translated_high = generator(source_high_band)
    low_loss = F.l1_loss(translated_low.float(), target_low_band.float())
    high_loss = F.l1_loss(translated_high.float(), target_high_band.float())
    return (low_loss + high_loss, low_loss, high_loss, source_low, source_high, target_low, target_high)
