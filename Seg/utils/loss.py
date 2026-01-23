
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        """
        初始化 Dice Loss
        :param smooth: 一个很小的常数，用于避免除零错误（默认值为 1e-5）
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        计算 Dice Loss
        :param pred: 预测结果，Tensor 类型，形状为 [batch_size, num_classes, H, W]
        :param target: 真实标签，Tensor 类型，形状为 [batch_size, num_classes, H, W]
        :return: 计算得到的 Dice Loss 值
        """
        # 将预测值和真实值从 [0, 1] 转为二值化
        # 假设预测值经过 sigmoid 激活并且在 [0, 1] 之间
        pred = torch.sigmoid(pred)

        # 平均每个像素位置的计算 Dice 系数
        intersection = torch.sum(pred * target)  # 交集
        union = torch.sum(pred) + torch.sum(target)  # 并集

        # 计算 Dice 系数并返回 Dice Loss
        dice = (2. * intersection + self.smooth) / (union + intersection + self.smooth)
        dice_loss = 1 - dice

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=5.0, reduction='mean'):
        """
        初始化 Focal Loss
        :param alpha: 平衡因子，通常设置为 0.25，调整正负类的比例
        :param gamma: 焦点因子，通常设置为 2，用于调整难分类样本的权重
        :param reduction: 损失的计算方式，'mean' 为平均损失，'sum' 为总损失，'none' 为不做任何汇总
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算 Focal Loss
        :param inputs: 预测值，形状为 [batch_size, 1, H, W] 或 [batch_size, H, W]
        :param targets: 真实标签，形状为 [batch_size, 1, H, W] 或 [batch_size, H, W]
        :return: 计算得到的 Focal Loss 值
        """
        # 将输入经过 sigmoid 激活，得到每个像素为目标类的概率值
        inputs = torch.sigmoid(inputs)

        # 计算 p_t（模型预测的目标类概率）
        p_t = inputs * targets + (1 - inputs) * (1 - targets)

        # 计算 Focal Loss 的核心部分
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)  # 1e-8 是避免 log(0)

        # 选择如何汇总损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        计算 IoU 损失
        y_true: 真实标签，Tensor 类型，大小为 [batch_size, 1, height, width] 或 [batch_size, height, width]
        y_pred: 模型的预测输出，Tensor 类型，大小为 [batch_size, 1, height, width] 或 [batch_size, height, width]

        返回：
        IoU 损失（值越小越好）
        """
        # 确保预测和真实标签都是二值化的
        inputs = torch.sigmoid(y_pred)  # 对预测值应用 sigmoid 函数，得到概率值

        # 计算交集（Intersection）
        intersection = torch.sum(y_true * inputs)

        # 计算并集（Union）
        union = torch.sum(y_true) + torch.sum(inputs) - intersection

        # 计算 IoU
        iou = intersection / (union + 1e-8)  # 添加一个小值防止除零

        # 返回 IoU 损失（1 - IoU）
        return 1 - iou


class BoundaryLoss(nn.Module):
    def __init__(self, alpha=1.0, batchsize=8):
        """
        Boundary Loss 实现，强调分割边界的精确度
        alpha: 权重因子，用于平衡边界损失和其他损失
        """
        super(BoundaryLoss, self).__init__()
        self.alpha = alpha
        self.batchsize = batchsize

    def forward(self, pred, target):
        """
        pred: 模型的预测结果，Tensor 形状 (B, C, H, W)
        target: 真实标签，Tensor 形状 (B, C, H, W)
        """
        # 计算边界：首先使用膨胀（dilation）得到分割的边界
        pred = torch.sigmoid(pred)  # 对预测结果应用 sigmoid 激活函数
        pred = (pred >= 0.5).float()

        target = target.float()  # 确保 target 是 float 类型

        # 计算边界
        pred_boundary = self.compute_boundary(pred)
        target_boundary = self.compute_boundary(target)

        # 计算边界损失
        loss = F.binary_cross_entropy(pred_boundary, target_boundary)
        return loss

    def compute_boundary(self, tensor):
        """
        计算二值图像的边界，使用膨胀操作和图像的差异来找到边界
        tensor: 输入的二值图像（预测或真实标签）
        """
        # 使用膨胀操作来标识边界
        kernel = torch.ones((self.batchsize, 1, 3, 3), device=tensor.device)  # 定义一个 3x3 的卷积核
        dilated = F.conv2d(tensor, kernel, padding=1).to(tensor.device)

        # 计算边界，膨胀后的图像减去原图
        boundary = dilated - tensor
        return boundary


def Distillation_loss(f1, f2):
    loss = 0.0
    assert len(f1) == len(f2)

    for i in range(len(f1)):
        loss = loss + F.mse_loss(f1[i], f2[i].detach())

    return loss


# def Distillation_loss(f1, f2, attention):
#     assert len(f1) == len(f2)
#     _, _, H, W = f1[0].size()
#
#     for i in range(len(f1)):
#         f1[i] = F.interpolate(f1[i], size=[H, W])
#         f2[i] = F.interpolate(f2[i], size=[H, W])
#
#     f1_tensor = torch.cat(f1, dim=1)
#     f2_tensor = torch.cat(f2, dim=1)
#     loss = F.mse_loss(f1_tensor, f2_tensor)
#
#     return loss


def kl_divergence(p, q):
    """
    计算KL散度，p为目标分布（mask），q为注意力图（attention map）

    参数：
    - p: 目标分布，形状为(B, H*W)，mask中为1的区域
    - q: 生成分布，形状为(B, H*W)，计算出来的attention map

    返回：
    - KL散度
    """
    # 为了避免计算log(0)，将q加一个很小的epsilon值
    epsilon = 1e-7
    q = torch.clamp(q, min=epsilon)

    # 计算KL散度：KL(p || q) = sum(p * log(p / q))
    kl_loss = torch.sum(p * torch.log(p / q), dim=-1)
    return torch.mean(kl_loss)


def js_divergence(p, q):
    """
    计算JS散度
    JS(P || Q) = 0.5 * (KL(P || M) + KL(Q || M))，其中 M = 0.5 * (P + Q)
    """
    # 计算均值分布 M
    m = 0.5 * (p + q)

    # 计算KL(P || M) 和 KL(Q || M)
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)

    # 计算JS散度
    js_loss = 0.5 * (kl_pm + kl_qm)
    return js_loss

class PixelSelfAttention(nn.Module):
    """
    计算像素级 self‑attention。
    输入  : B × C × H × W
    输出  : out  -> B × C × H × W  (加权后特征)
            attn -> B × N × N      (像素对像素 attention，N = H*W)
    """
    def __init__(self, in_channels: int, embed_dim: int = None, proj=True, down=2):
        super().__init__()
        # 如果不指定 embed_dim，就默认等于输入通道数
        embed_dim = embed_dim or in_channels

        # 用 1×1 卷积代替线性层，保持空间尺寸不变
        if proj:
            self.q_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False)
            self.k_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False)
            # self.v_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, bias=False)
        else:
            self.q_proj = nn.Identity()
            self.k_proj = nn.Identity()

        # 缩放因子，避免点积过大
        self.scale = embed_dim ** -0.5

        # 下采样倍数
        self.down = down

        # 可选：把输出再映射回原通道数
        # self.out_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (B, C, H, W)
        Returns:
            out : (B, C, H, W)  – attention 加权后的特征
            attn: (B, N, N)    – 像素‑像素 attention map，可视化或做后续约束
        """
        # 下采样
        x = F.avg_pool2d(x, (self.down, self.down))

        B, C, H, W = x.shape
        N = H * W                       # token 数

        # 1) 计算 Q, K, V
        q = self.q_proj(x)              # (B, d, H, W)
        k = self.k_proj(x)
        # v = self.v_proj(x)


        # 2) 拉平成 (B, N, d) 便于批量矩阵乘
        q = q.flatten(2).transpose(1, 2)  # (B, N, d)
        k = k.flatten(2)                  # (B, d, N)
        # v = v.flatten(2).transpose(1, 2)  # (B, N, d)

        # 3) 计算 attention 分数 & softmax
        attn_scores = torch.bmm(q, k) * self.scale   # (B, N, N)
        attn = F.softmax(attn_scores, dim=-1)

        # # 4) 加权求和得到输出，再 reshape 回原形状
        # out = torch.bmm(attn, v)                     # (B, N, d)
        # out = out.transpose(1, 2).view(B, -1, H, W)  # (B, d, H, W)
        # out = self.out_proj(out)                     # (B, C, H, W)

        return attn  # , out


# ----------- 使用示例 -------------
if __name__ == "__main__":
    B, C, H, W = 2, 1, 240, 240                # 例如 B=2, 单通道 240×240 图
    x = torch.randn(B, C, H, W).cuda()
    y = torch.randn(B, C, H, W).cuda()

    psa = PixelSelfAttention(in_channels=C).cuda()
    psa_y = PixelSelfAttention(in_channels=C, proj=False).cuda()
    attn_map = psa(x)

    # print("输出特征尺寸 :", out_feat.shape)     # (2, 1, 240, 240)
    print("Attention 尺寸:", attn_map.shape)    # (2, 57600, 57600)
    loss = js_divergence(torch.flatten(psa_y(y), 1), torch.flatten(attn_map, 1))
    print(loss)
