import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapLoss(nn.Module):

    def __init__(self, hard_bg_fraction, hard_bg_min_k, hard_bg_max_k, min_peak_pixels, fg_weight=1.0, bg_weight=0.25, fg_gamma=1.0, bg_gamma=4.0, kl_weight=0.03, peak_threshold=0.7, peak_fallback_threshold=0.5, peak_weight=0.9, hard_bg_threshold=0.05, hard_bg_weight=0.45, rank_weight=0.3, rank_margin=2.0, mass_weight=0.02, mse_weight=0.05, eps=1e-06):
        super().__init__()
        required = {'hard_bg_fraction': hard_bg_fraction, 'hard_bg_min_k': hard_bg_min_k, 'hard_bg_max_k': hard_bg_max_k, 'min_peak_pixels': min_peak_pixels}
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError('Missing implementation-specific detection parameters: ' + ', '.join(missing))
        self.hard_bg_fraction = float(hard_bg_fraction)
        self.hard_bg_min_k = int(hard_bg_min_k)
        self.hard_bg_max_k = int(hard_bg_max_k)
        self.min_peak_pixels = int(min_peak_pixels)
        self.fg_weight = float(fg_weight)
        self.bg_weight = float(bg_weight)
        self.fg_gamma = float(fg_gamma)
        self.bg_gamma = float(bg_gamma)
        self.kl_weight = float(kl_weight)
        self.peak_threshold = float(peak_threshold)
        self.peak_fallback_threshold = float(peak_fallback_threshold)
        self.peak_weight = float(peak_weight)
        self.hard_bg_threshold = float(hard_bg_threshold)
        self.hard_bg_weight = float(hard_bg_weight)
        self.rank_weight = float(rank_weight)
        self.rank_margin = float(rank_margin)
        self.mass_weight = float(mass_weight)
        self.mse_weight = float(mse_weight)
        self.eps = float(eps)

    def _peak_mask(self, gt_flat):
        mask = gt_flat >= self.peak_threshold
        if int(mask.sum().item()) >= self.min_peak_pixels:
            return mask
        mask = gt_flat >= self.peak_fallback_threshold
        if int(mask.sum().item()) >= self.min_peak_pixels:
            return mask
        positive = gt_flat > 0
        count = int(positive.sum().item())
        if count == 0:
            return mask
        k = min(max(self.min_peak_pixels, 1), count)
        positive_idx = torch.nonzero(positive, as_tuple=False).squeeze(1)
        local_idx = torch.topk(gt_flat[positive], k=k, largest=True).indices
        mask = torch.zeros_like(gt_flat, dtype=torch.bool)
        mask[positive_idx[local_idx]] = True
        return mask

    def forward(self, logits, gt):
        logits = logits.float()
        gt = gt.float().clamp(0.0, 1.0)
        if logits.shape != gt.shape:
            raise ValueError(f'logits and gt shape mismatch: {logits.shape} vs {gt.shape}')
        pred = torch.sigmoid(logits)
        logits_flat = logits.flatten(1)
        pred_flat = pred.flatten(1)
        gt_flat = gt.flatten(1)
        bce_map = F.binary_cross_entropy_with_logits(logits, gt, reduction='none').flatten(1)
        fg = gt.pow(self.fg_gamma).flatten(1)
        bg = (1.0 - gt).pow(self.bg_gamma).flatten(1)
        fg_loss = (fg * bce_map).sum(1) / fg.sum(1).clamp_min(self.eps)
        bg_loss = (bg * bce_map).sum(1) / bg.sum(1).clamp_min(self.eps)
        balanced_bce = (self.fg_weight * fg_loss + self.bg_weight * bg_loss).mean()
        gt_sum = gt_flat.sum(1, keepdim=True)
        valid = gt_sum.squeeze(1) > self.eps
        if valid.any():
            gt_dist = gt_flat[valid] / gt_sum[valid].clamp_min(self.eps)
            pred_valid = pred_flat[valid].clamp_min(self.eps)
            pred_dist = pred_valid / pred_valid.sum(1, keepdim=True).clamp_min(self.eps)
            kl = (gt_dist * (torch.log(gt_dist + self.eps) - torch.log(pred_dist + self.eps))).sum(1).mean()
        else:
            kl = logits.sum() * 0.0
        peak_losses = []
        hard_bg_losses = []
        rank_losses = []
        for i in range(logits.shape[0]):
            peak_mask = self._peak_mask(gt_flat[i])
            if peak_mask.any():
                peak_losses.append(-F.logsigmoid(logits_flat[i][peak_mask]).mean())
            bg_mask = gt_flat[i] < self.hard_bg_threshold
            if not bg_mask.any():
                continue
            bg_logits = logits_flat[i][bg_mask]
            k = int(self.hard_bg_fraction * bg_logits.numel())
            k = max(k, self.hard_bg_min_k)
            k = min(k, self.hard_bg_max_k, bg_logits.numel())
            if k <= 0:
                continue
            hard_logits = torch.topk(bg_logits, k=k, largest=True).values
            hard_bg_losses.append(-F.logsigmoid(-hard_logits).mean())
            if peak_mask.any():
                peak_logits = logits_flat[i][peak_mask]
                peak_gt = gt_flat[i][peak_mask]
                anchor = (peak_logits * peak_gt).sum() / peak_gt.sum().clamp_min(self.eps)
                rank_losses.append(F.softplus(hard_logits - anchor + self.rank_margin).mean())
        zero = logits.sum() * 0.0
        peak_loss = torch.stack(peak_losses).mean() if peak_losses else zero
        hard_bg_loss = torch.stack(hard_bg_losses).mean() if hard_bg_losses else zero
        rank_loss = torch.stack(rank_losses).mean() if rank_losses else zero
        mass_loss = F.smooth_l1_loss(torch.log(pred_flat.sum(1).clamp_min(self.eps)), torch.log(gt_flat.sum(1).clamp_min(self.eps)), reduction='mean')
        mse_loss = F.mse_loss(pred, gt, reduction='mean')
        total = balanced_bce + self.kl_weight * kl + self.peak_weight * peak_loss + self.hard_bg_weight * hard_bg_loss + self.rank_weight * rank_loss + self.mass_weight * mass_loss + self.mse_weight * mse_loss
        if not torch.isfinite(total):
            raise RuntimeError('Non-finite heatmap loss')
        return (total, {'focal': balanced_bce.detach(), 'weighted_mse': mse_loss.detach(), 'balanced_bce': balanced_bce.detach(), 'dist_kl': kl.detach(), 'peak_pos': peak_loss.detach(), 'hard_bg': hard_bg_loss.detach(), 'rank': rank_loss.detach(), 'mass': mass_loss.detach(), 'mse': mse_loss.detach()})
