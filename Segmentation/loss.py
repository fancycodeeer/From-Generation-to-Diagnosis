import torch
import torch.nn as nn
import torch.nn.functional as F

class ComboSegLoss(nn.Module):

    def __init__(self, bce_weight: float=0.25, dice_weight: float=1.0, iou_weight: float=1.0, focal_weight: float=1.75, focal_alpha: float=0.25, focal_gamma: float=5.0, pos_weight: float=0.0, eps: float=1e-06):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.iou_weight = float(iou_weight)
        self.focal_weight = float(focal_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.eps = float(eps)
        if pos_weight and pos_weight > 0:
            self.register_buffer('pos_weight', torch.tensor([float(pos_weight)], dtype=torch.float32))
        else:
            self.pos_weight = None

    def per_sample(self, logits: torch.Tensor, targets: torch.Tensor):
        bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
        bce = bce_map.flatten(1).mean(dim=1)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        focal_map = alpha_t * (1.0 - pt).pow(self.focal_gamma) * bce_map
        focal = focal_map.flatten(1).mean(dim=1)
        probs_flat = probs.flatten(1)
        targets_flat = targets.flatten(1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        prob_sum = probs_flat.sum(dim=1)
        target_sum = targets_flat.sum(dim=1)
        dice = 1.0 - (2.0 * intersection + self.eps) / (prob_sum + target_sum + self.eps)
        union = prob_sum + target_sum - intersection
        iou = 1.0 - (intersection + self.eps) / (union + self.eps)
        total = self.bce_weight * bce + self.focal_weight * focal + self.iou_weight * iou + self.dice_weight * dice
        return (total, {'loss_bce': bce, 'loss_focal': focal, 'loss_iou': iou, 'loss_dice': dice})

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        total, components = self.per_sample(logits, targets)
        return (total.mean(), {key: value.mean() for key, value in components.items()})
