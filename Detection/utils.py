import torch
import torch.nn.functional as F
import torch.nn as nn

# ------------------------- Decoding -------------------------
def _nms(heat, kernel=3):
    pad = (kernel-1)//2
    hmax = F.max_pool2d(heat,(kernel,kernel),1,pad)
    return heat*(hmax==heat)

def topk(scores,K=2):
    B,C,H,W = scores.shape
    topk_scores, topk_inds = torch.topk(scores.view(B,C,-1),K)
    topk_inds = topk_inds%(H*W)
    ys = (topk_inds//W).float(); xs = (topk_inds%W).float()
    topk_scores, ind = torch.topk(topk_scores.view(B,-1),K)
    clses = (ind//K).int()
    inds = topk_inds.view(B,-1)[torch.arange(B)[:,None],ind]
    ys = ys.view(B,-1)[torch.arange(B)[:,None],ind]
    xs = xs.view(B,-1)[torch.arange(B)[:,None],ind]
    return topk_scores, inds, clses, ys, xs

def decode(hm, wh, reg, K=2):
    hm = _nms(hm)
    scores, inds, clses, ys, xs = topk(hm,K)
    wh = wh.permute(0,2,3,1).reshape(hm.size(0),-1,2)
    reg = reg.permute(0,2,3,1).reshape(hm.size(0),-1,2)
    wh = wh[torch.arange(hm.size(0))[:,None],inds]
    reg= reg[torch.arange(hm.size(0))[:,None],inds]
    xs = xs+reg[...,0]; ys=ys+reg[...,1]
    half_w,half_h = wh[...,0]/2,wh[...,1]/2
    bboxes = torch.stack([xs-half_w, ys-half_h, xs+half_w, ys+half_h],-1)
    return bboxes, scores, clses


def top1_peak_coords(hm, thresh=0.01, kernel_size=51, stride=1, padding=25):
    """
    输入:
      hm: Tensor of shape (B, C, H, W)
    返回:
      coords: Tensor of shape (B, C, 2)  每个样本每个通道的 (x, y) 坐标
      vals:   Tensor of shape (B, C)     该点的热图值
    """
    B, C, H, W = hm.shape
    # 1) 局部极大值检测
    pooled = F.max_pool2d(hm, kernel_size=kernel_size, stride=stride, padding=padding)
    peaks = (hm == pooled) & (hm > thresh)      # bool mask

    coords = torch.zeros((B, C, 2), dtype=torch.long, device=hm.device)
    vals   = torch.zeros((B, C),    dtype=hm.dtype, device=hm.device)

    for b in range(B):
        for c in range(C):
            # 找到峰的位置索引列表 (n_peaks, 2)
            idxs = torch.nonzero(peaks[b, c], as_tuple=False)  # 每行是 [y, x]
            if idxs.numel() == 0:
                # 如果没有峰，则保留 (0,0)，值为0
                continue
            # 这些峰对应的热图值
            peak_vals = hm[b, c][idxs[:, 0], idxs[:, 1]]
            # 在这些峰中取数值最大的那个
            max_i = torch.argmax(peak_vals)
            y, x = idxs[max_i]
            coords[b, c, 0] = x
            coords[b, c, 1] = y
            vals[b, c] = peak_vals[max_i]
    return coords, vals


def Distillation_loss(f1, f2):
    loss = 0.0
    assert len(f1) == len(f2)

    for i in range(len(f1)):
        loss = loss + F.mse_loss(f1[i], f2[i])

    return loss


def DiceLoss(pred, target):
    """
    计算 Dice Loss
    :param pred: 预测结果，Tensor 类型，形状为 [batch_size, num_classes, H, W]
    :param target: 真实标签，Tensor 类型，形状为 [batch_size, num_classes, H, W]
    :return: 计算得到的 Dice Loss 值
    """
    # 将预测值和真实值从 [0, 1] 转为二值化
    # 假设预测值经过 sigmoid 激活并且在 [0, 1] 之间
    # pred = torch.sigmoid(pred)

    # 平均每个像素位置的计算 Dice 系数
    intersection = torch.sum(pred * target)  # 交集
    union = torch.sum(pred) + torch.sum(target)  # 并集

    # 计算 Dice 系数并返回 Dice Loss
    dice = (2. * intersection + 1e-8) / (union + intersection + 1e-8)
    dice_loss = 1 - dice

    return dice_loss


def FocalLoss(inputs, targets):
    """
    计算 Focal Loss
    :param inputs: 预测值，形状为 [batch_size, 1, H, W] 或 [batch_size, H, W]
    :param targets: 真实标签，形状为 [batch_size, 1, H, W] 或 [batch_size, H, W]
    :return: 计算得到的 Focal Loss 值
    """
    # 将输入经过 sigmoid 激活，得到每个像素为目标类的概率值
    # inputs = torch.sigmoid(inputs)

    # 计算 p_t（模型预测的目标类概率）
    p_t = inputs * targets + (1 - inputs) * (1 - targets)

    # 计算 Focal Loss 的核心部分
    loss = -0.25 * (1 - p_t) ** 5.0 * torch.log(p_t + 1e-8)  # 1e-8 是避免 log(0)

    return loss.mean()


def IoULoss(y_pred, y_true):
    """
    计算 IoU 损失
    y_true: 真实标签，Tensor 类型，大小为 [batch_size, 1, height, width] 或 [batch_size, height, width]
    y_pred: 模型的预测输出，Tensor 类型，大小为 [batch_size, 1, height, width] 或 [batch_size, height, width]

    返回：
    IoU 损失（值越小越好）
    """
    # 确保预测和真实标签都是二值化的
    # inputs = torch.sigmoid(y_pred)  # 对预测值应用 sigmoid 函数，得到概率值
    inputs = y_pred

    # 计算交集（Intersection）
    intersection = torch.sum(y_true * inputs)

    # 计算并集（Union）
    union = torch.sum(y_true) + torch.sum(inputs) - intersection

    # 计算 IoU
    iou = intersection / (union + 1e-8)  # 添加一个小值防止除零

    # 返回 IoU 损失（1 - IoU）
    return 1 - iou
