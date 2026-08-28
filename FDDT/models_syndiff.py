from dataclasses import dataclass
from typing import Dict, Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, 'module') else model

def set_requires_grad(nets: Iterable[nn.Module], requires_grad: bool) -> None:
    for net in nets:
        if net is None:
            continue
        for p in net.parameters():
            p.requires_grad = requires_grad

def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def model_state_dict(model: nn.Module) -> Dict:
    return unwrap_model(model).state_dict()

def load_model_state(model: nn.Module, state: Dict, strict: bool=True) -> None:
    unwrap_model(model).load_state_dict(state, strict=strict)

def ddp_wrap_models(models, local_rank: int, find_unused_parameters: bool=False):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return models
    for name in models.__dataclass_fields__.keys():
        net = getattr(models, name)
        setattr(models, name, torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=find_unused_parameters, broadcast_buffers=False))
    return models

def make_group_norm(channels: int, max_groups: int=8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        t = t.float()
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * torch.arange(half, device=device).float() / max(half - 1, 1))
        emb = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class TimestepLatentMLP(nn.Module):

    def __init__(self, time_dim: int=256, z_dim: int=100, out_dim: int=256):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.z_dim = z_dim
        self.net = nn.Sequential(nn.Linear(time_dim + z_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t: torch.Tensor, z: Optional[torch.Tensor]=None) -> torch.Tensor:
        te = self.time(t)
        if z is None:
            z = torch.zeros(t.shape[0], self.z_dim, device=t.device, dtype=te.dtype)
        else:
            z = z.to(dtype=te.dtype)
        return self.net(torch.cat([te, z], dim=1))

class FiLMResBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float=0.0):
        super().__init__()
        self.norm1 = make_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_ch * 2))
        self.norm2 = make_group_norm(out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        scale_shift = self.emb(e).to(dtype=h.dtype)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)

class Downsample(nn.Module):

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

class Upsample(nn.Module):

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        return self.op(x)

class SelfAttention2d(nn.Module):

    def __init__(self, ch: int, num_heads: int=4):
        super().__init__()
        self.ch = ch
        self.num_heads = num_heads
        self.norm = make_group_norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        head_dim = c // self.num_heads
        if c % self.num_heads != 0:
            return residual
        q = q.view(b, self.num_heads, head_dim, h * w)
        k = k.view(b, self.num_heads, head_dim, h * w)
        v = v.view(b, self.num_heads, head_dim, h * w)
        scale = head_dim ** (-0.5)
        attn = torch.einsum('bhcn,bhcm->bhnm', q * scale, k)
        attn = torch.softmax(attn.float(), dim=-1).to(dtype=v.dtype)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return residual + out

class EnhancedConditionalUNetGenerator(nn.Module):

    def __init__(self, in_nc: int=2, out_nc: int=1, base_ch: int=64, z_dim: int=100, emb_dim: int=256, dropout: float=0.0, use_attention: bool=True):
        super().__init__()
        self.emb = TimestepLatentMLP(time_dim=emb_dim, z_dim=z_dim, out_dim=emb_dim)
        ch1 = base_ch
        ch2 = base_ch * 2
        ch3 = base_ch * 4
        ch4 = base_ch * 8
        self.in_conv = nn.Conv2d(in_nc, ch1, kernel_size=3, padding=1)
        self.enc1 = nn.ModuleList([FiLMResBlock(ch1, ch1, emb_dim, dropout), FiLMResBlock(ch1, ch1, emb_dim, dropout)])
        self.down1 = Downsample(ch1)
        self.enc2 = nn.ModuleList([FiLMResBlock(ch1, ch2, emb_dim, dropout), FiLMResBlock(ch2, ch2, emb_dim, dropout)])
        self.down2 = Downsample(ch2)
        self.enc3 = nn.ModuleList([FiLMResBlock(ch2, ch3, emb_dim, dropout), FiLMResBlock(ch3, ch3, emb_dim, dropout)])
        self.down3 = Downsample(ch3)
        self.mid_in = FiLMResBlock(ch3, ch4, emb_dim, dropout)
        self.mid_attn = SelfAttention2d(ch4, num_heads=4) if use_attention else nn.Identity()
        self.mid_out = FiLMResBlock(ch4, ch4, emb_dim, dropout)
        self.up3 = Upsample(ch4)
        self.dec3 = nn.ModuleList([FiLMResBlock(ch4 + ch3, ch3, emb_dim, dropout), FiLMResBlock(ch3, ch3, emb_dim, dropout)])
        self.up2 = Upsample(ch3)
        self.dec2 = nn.ModuleList([FiLMResBlock(ch3 + ch2, ch2, emb_dim, dropout), FiLMResBlock(ch2, ch2, emb_dim, dropout)])
        self.up1 = Upsample(ch2)
        self.dec1 = nn.ModuleList([FiLMResBlock(ch2 + ch1, ch1, emb_dim, dropout), FiLMResBlock(ch1, ch1, emb_dim, dropout)])
        self.out = nn.Sequential(make_group_norm(ch1), nn.SiLU(), nn.Conv2d(ch1, ch1, kernel_size=3, padding=1), nn.SiLU(), nn.Conv2d(ch1, out_nc, kernel_size=3, padding=1), nn.Tanh())
        self.apply(init_weights)

    def _run_blocks(self, blocks: nn.ModuleList, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            x = block(x, e)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: Optional[torch.Tensor]=None) -> torch.Tensor:
        e = self.emb(t, z)
        h = self.in_conv(x)
        s1 = self._run_blocks(self.enc1, h, e)
        h = self.down1(s1)
        s2 = self._run_blocks(self.enc2, h, e)
        h = self.down2(s2)
        s3 = self._run_blocks(self.enc3, h, e)
        h = self.down3(s3)
        h = self.mid_in(h, e)
        h = self.mid_attn(h)
        h = self.mid_out(h, e)
        h = self.up3(h)
        if h.shape[-2:] != s3.shape[-2:]:
            h = F.interpolate(h, size=s3.shape[-2:], mode='nearest')
        h = torch.cat([h, s3], dim=1)
        h = self._run_blocks(self.dec3, h, e)
        h = self.up2(h)
        if h.shape[-2:] != s2.shape[-2:]:
            h = F.interpolate(h, size=s2.shape[-2:], mode='nearest')
        h = torch.cat([h, s2], dim=1)
        h = self._run_blocks(self.dec2, h, e)
        h = self.up1(h)
        if h.shape[-2:] != s1.shape[-2:]:
            h = F.interpolate(h, size=s1.shape[-2:], mode='nearest')
        h = torch.cat([h, s1], dim=1)
        h = self._run_blocks(self.dec1, h, e)
        return self.out(h)

class ResnetBlock(nn.Module):

    def __init__(self, dim: int, dropout: float=0.0):
        super().__init__()
        layers = [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), nn.InstanceNorm2d(dim, affine=False), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), nn.InstanceNorm2d(dim, affine=False)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class ResnetGenerator(nn.Module):

    def __init__(self, input_nc: int=1, output_nc: int=1, ngf: int=64, n_blocks: int=9, dropout: float=0.0):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7), nn.InstanceNorm2d(ngf, affine=False), nn.ReLU()]
        mult = 1
        for _ in range(2):
            layers += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), nn.InstanceNorm2d(ngf * mult * 2, affine=False), nn.ReLU()]
            mult *= 2
        for _ in range(n_blocks):
            layers.append(ResnetBlock(ngf * mult, dropout=dropout))
        for _ in range(2):
            layers += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(ngf * mult // 2, affine=False), nn.ReLU()]
            mult //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7), nn.Tanh()]
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class PatchDiscriminator(nn.Module):

    def __init__(self, input_nc: int=1, ndf: int=64, n_layers: int=3, use_spectral_norm: bool=True):
        super().__init__()

        def conv(in_c, out_c, k=4, s=2, p=1):
            layer = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            return spectral_norm(layer) if use_spectral_norm else layer
        layers = [conv(input_nc, ndf), nn.LeakyReLU(0.2)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [conv(ndf * nf_mult_prev, ndf * nf_mult), nn.InstanceNorm2d(ndf * nf_mult, affine=False), nn.LeakyReLU(0.2)]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [conv(ndf * nf_mult_prev, ndf * nf_mult, s=1), nn.InstanceNorm2d(ndf * nf_mult, affine=False), nn.LeakyReLU(0.2), conv(ndf * nf_mult, 1, s=1)]
        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class DiffusionDiscriminator(nn.Module):

    def __init__(self, nc: int=2, ndf: int=64, t_emb_dim: int=256, use_spectral_norm: bool=True):
        super().__init__()
        self.time = nn.Sequential(SinusoidalTimeEmbedding(t_emb_dim), nn.Linear(t_emb_dim, t_emb_dim), nn.SiLU(), nn.Linear(t_emb_dim, ndf * 4))

        def conv(in_c, out_c, k=4, s=2, p=1):
            layer = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            return spectral_norm(layer) if use_spectral_norm else layer
        self.c1 = conv(nc, ndf)
        self.c2 = conv(ndf, ndf * 2)
        self.n2 = nn.InstanceNorm2d(ndf * 2, affine=False)
        self.c3 = conv(ndf * 2, ndf * 4)
        self.n3 = nn.InstanceNorm2d(ndf * 4, affine=False)
        self.c4 = conv(ndf * 4, ndf * 4, s=1)
        self.n4 = nn.InstanceNorm2d(ndf * 4, affine=False)
        self.out = conv(ndf * 4, 1, s=1)
        self.act = nn.LeakyReLU(0.2)
        self.apply(init_weights)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_tp1: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_t, x_tp1], dim=1)
        h = self.act(self.c1(x))
        h = self.act(self.n2(self.c2(h)))
        h = self.act(self.n3(self.c3(h)))
        te = self.time(t).to(dtype=h.dtype)[:, :, None, None]
        h = h + te
        h = self.act(self.n4(self.c4(h)))
        return self.out(h)

@dataclass
class SynDiffCoreBundle:
    gen_diffusive_1: nn.Module
    gen_diffusive_2: nn.Module
    gen_non_diffusive_1to2: nn.Module
    gen_non_diffusive_2to1: nn.Module
    disc_diffusive_1: nn.Module
    disc_diffusive_2: nn.Module
    disc_cycle_1: nn.Module
    disc_cycle_2: nn.Module

def build_syndiff_core_models(args, device: torch.device) -> SynDiffCoreBundle:
    if args.input_nc != 1 or args.output_nc != 1:
        raise ValueError('This SynDiff-core implementation is configured for single-channel medical translation.')
    gen_diffusive_1 = EnhancedConditionalUNetGenerator(in_nc=args.input_nc + args.output_nc, out_nc=args.input_nc, base_ch=args.diff_base_ch, z_dim=args.nz, emb_dim=args.t_emb_dim, dropout=0.0, use_attention=True).to(device)
    gen_diffusive_2 = EnhancedConditionalUNetGenerator(in_nc=args.input_nc + args.output_nc, out_nc=args.output_nc, base_ch=args.diff_base_ch, z_dim=args.nz, emb_dim=args.t_emb_dim, dropout=0.0, use_attention=True).to(device)
    n_cycle_blocks = max(int(args.resnet_blocks), 9)
    gen_non_diffusive_1to2 = ResnetGenerator(input_nc=args.input_nc, output_nc=args.output_nc, ngf=args.ngf, n_blocks=n_cycle_blocks, dropout=0.0).to(device)
    gen_non_diffusive_2to1 = ResnetGenerator(input_nc=args.output_nc, output_nc=args.input_nc, ngf=args.ngf, n_blocks=n_cycle_blocks, dropout=0.0).to(device)
    disc_diffusive_1 = DiffusionDiscriminator(nc=args.input_nc * 2, ndf=args.ndf, t_emb_dim=args.t_emb_dim, use_spectral_norm=args.spectral_norm).to(device)
    disc_diffusive_2 = DiffusionDiscriminator(nc=args.output_nc * 2, ndf=args.ndf, t_emb_dim=args.t_emb_dim, use_spectral_norm=args.spectral_norm).to(device)
    disc_cycle_1 = PatchDiscriminator(input_nc=args.input_nc, ndf=args.ndf, n_layers=3, use_spectral_norm=args.spectral_norm).to(device)
    disc_cycle_2 = PatchDiscriminator(input_nc=args.output_nc, ndf=args.ndf, n_layers=3, use_spectral_norm=args.spectral_norm).to(device)
    return SynDiffCoreBundle(gen_diffusive_1=gen_diffusive_1, gen_diffusive_2=gen_diffusive_2, gen_non_diffusive_1to2=gen_non_diffusive_1to2, gen_non_diffusive_2to1=gen_non_diffusive_2to1, disc_diffusive_1=disc_diffusive_1, disc_diffusive_2=disc_diffusive_2, disc_cycle_1=disc_cycle_1, disc_cycle_2=disc_cycle_2)

class EMAHelper:

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        model = unwrap_model(model)
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.decay <= 0:
            return
        model = unwrap_model(model)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        if self.decay <= 0:
            return
        model = unwrap_model(model)
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self.decay <= 0:
            return
        model = unwrap_model(model)
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self) -> Dict:
        return {'decay': self.decay, 'shadow': {k: v.detach().cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Optional[Dict], device: torch.device) -> None:
        if not state:
            return
        self.decay = float(state.get('decay', self.decay))
        self.shadow = {k: v.to(device=device).detach().clone() for k, v in state.get('shadow', {}).items()}

def build_ema_helpers(args, models: SynDiffCoreBundle) -> Dict[str, EMAHelper]:
    if not args.use_ema:
        return {}
    return {'gen_diffusive_1': EMAHelper(models.gen_diffusive_1, args.ema_decay), 'gen_diffusive_2': EMAHelper(models.gen_diffusive_2, args.ema_decay), 'gen_non_diffusive_1to2': EMAHelper(models.gen_non_diffusive_1to2, args.ema_decay), 'gen_non_diffusive_2to1': EMAHelper(models.gen_non_diffusive_2to1, args.ema_decay)}

@torch.no_grad()
def update_ema_helpers(ema: Dict[str, EMAHelper], models: SynDiffCoreBundle) -> None:
    if not ema:
        return
    ema['gen_diffusive_1'].update(models.gen_diffusive_1)
    ema['gen_diffusive_2'].update(models.gen_diffusive_2)
    ema['gen_non_diffusive_1to2'].update(models.gen_non_diffusive_1to2)
    ema['gen_non_diffusive_2to1'].update(models.gen_non_diffusive_2to1)

@torch.no_grad()
def apply_ema_to_generators(ema: Dict[str, EMAHelper], models: SynDiffCoreBundle) -> None:
    if not ema:
        return
    ema['gen_diffusive_1'].apply_shadow(models.gen_diffusive_1)
    ema['gen_diffusive_2'].apply_shadow(models.gen_diffusive_2)
    ema['gen_non_diffusive_1to2'].apply_shadow(models.gen_non_diffusive_1to2)
    ema['gen_non_diffusive_2to1'].apply_shadow(models.gen_non_diffusive_2to1)

@torch.no_grad()
def restore_generators_from_ema(ema: Dict[str, EMAHelper], models: SynDiffCoreBundle) -> None:
    if not ema:
        return
    ema['gen_diffusive_1'].restore(models.gen_diffusive_1)
    ema['gen_diffusive_2'].restore(models.gen_diffusive_2)
    ema['gen_non_diffusive_1to2'].restore(models.gen_non_diffusive_1to2)
    ema['gen_non_diffusive_2to1'].restore(models.gen_non_diffusive_2to1)
