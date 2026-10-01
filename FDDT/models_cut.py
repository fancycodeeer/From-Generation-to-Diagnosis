import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ResnetBlock, get_norm_layer

class ResnetGeneratorCUT(nn.Module):

    def __init__(self, input_nc: int, output_nc: int, ngf: int=64, norm_layer=None, use_dropout: bool=False, n_blocks: int=9):
        super().__init__()
        if norm_layer is None:
            norm_layer = get_norm_layer('instance')
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True), norm_layer(ngf), nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True), norm_layer(ngf * mult * 2), nn.ReLU(True)]
        mult = 2 ** n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout)]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.ModuleList(model)

    def forward(self, x: torch.Tensor, layers=None, encode_only: bool=False):
        if layers is None:
            layers = []
        if isinstance(layers, str):
            layers = [int(x) for x in layers.split(',') if x.strip()]
        feats = []
        out = x
        if encode_only:
            if len(layers) == 0:
                raise ValueError('encode_only=True requires non-empty layers.')
            last_layer = max(layers)
        else:
            last_layer = len(self.model) - 1
        for i, layer in enumerate(self.model):
            out = layer(out)
            if i in layers:
                feats.append(out)
            if encode_only and i >= last_layer:
                return feats
        if encode_only:
            return feats
        return out

class PatchSampleF(nn.Module):

    def __init__(self, use_mlp: bool=True, nc: int=256):
        super().__init__()
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.mlps = nn.ModuleList()

    def create_mlp(self, feats):
        if self.mlp_init:
            return
        self.mlps = nn.ModuleList()
        for feat in feats:
            input_nc = int(feat.shape[1])
            mlp = nn.Sequential(nn.Linear(input_nc, self.nc), nn.ReLU(True), nn.Linear(self.nc, self.nc))
            self.mlps.append(mlp)
        self.mlp_init = True

    def forward(self, feats, num_patches: int=256, patch_ids=None):
        if self.use_mlp and (not self.mlp_init):
            self.create_mlp(feats)
            self.to(feats[0].device)
        return_feats = []
        return_ids = []
        for feat_id, feat in enumerate(feats):
            B, C, H, W = feat.shape
            feat_reshape = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id].to(feat.device)
                else:
                    patch_id = torch.randperm(H * W, device=feat.device)[:min(num_patches, H * W)]
                x_sample = feat_reshape[:, patch_id, :]
            else:
                patch_id = torch.arange(H * W, device=feat.device)
                x_sample = feat_reshape
            if self.use_mlp:
                x_sample = self.mlps[feat_id](x_sample.reshape(-1, C))
                x_sample = x_sample.reshape(B, -1, self.nc)
            x_sample = F.normalize(x_sample, dim=-1)
            return_feats.append(x_sample)
            return_ids.append(patch_id)
        return (return_feats, return_ids)

class PatchNCELoss(nn.Module):

    def __init__(self, temperature: float=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        B, S, C = feat_q.shape
        feat_q = F.normalize(feat_q, dim=-1)
        feat_k = F.normalize(feat_k, dim=-1)
        l_pos = torch.sum(feat_q * feat_k, dim=-1, keepdim=True)
        l_neg = torch.bmm(feat_q, feat_k.transpose(1, 2))
        diagonal = torch.eye(S, device=feat_q.device, dtype=torch.bool).unsqueeze(0)
        l_neg = l_neg.masked_fill(diagonal, -10.0)
        logits = torch.cat([l_pos, l_neg], dim=2)
        logits = logits / self.temperature
        targets = torch.zeros(B * S, dtype=torch.long, device=feat_q.device)
        loss = self.cross_entropy_loss(logits.reshape(B * S, S + 1), targets)
        return loss
