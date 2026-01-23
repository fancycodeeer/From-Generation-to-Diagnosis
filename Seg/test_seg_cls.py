#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import argparse
from models import UNet, Generator, CrossModalAttentionNetwork, AttentionFusionNet
import torchvision.transforms as transforms
from utils import Fourier_trans, make_mask_, make_distance_mask_
from torchvision.utils import save_image
from utils.loss import PixelSelfAttention
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

# --------------------------
# 辅助函数
# ---------------------------
Gray = transforms.Compose([transforms.Grayscale()])
def load_image(image_path):
    """
    读取图像并预处理：
    - 读取 BGR 图像并转换为 RGB
    - 归一化到 [0, 1]
    - 转换为 Tensor 格式，并调整为 [C, H, W]
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("图像读取失败: {}".format(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = torch.from_numpy(image.transpose((2, 0, 1))).float()
    image = Gray(image)
    return image

def save_mask(mask, output_path):
    """
    保存二值分割结果：
    - mask 为 numpy 数组（值在 0 和 1 之间），乘以 255 并转换为 uint8 保存
    """
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask)

# ---------- ②  全局平均 / 最大 ----------
def vis_global(attn: torch.Tensor, h: int, w: int, b: int = 0,
               mode='mean', cmap='hot', save=None):
    """
    mode = 'mean' | 'max'  对所有 Query 聚合
    """
    if mode == 'mean':
        A = attn[b].mean(0)         # (N,)
    else:
        A = attn[b].max(0).values
    A = A.view(h, w).float().cpu()
    plt.figure(figsize=(4, 4))
    plt.imshow(A, cmap=cmap)
    plt.colorbar(fraction=0.046)
    plt.title(f'Global {mode}')
    plt.axis('off')
    if save: plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()

def show_full_matrix(attn, idx=0, vmax=None):
    """
    显示第 idx 个样本的完整 attention 矩阵 (N×N)
    """
    A = attn[idx].detach().cpu()          # (N, N)
    vmax = vmax or A.max().item()
    plt.figure(figsize=(4, 4))
    sns.heatmap(A.float().cpu(), vmin=0, vmax=vmax, cmap='viridis')
    plt.title(f'Attention Matrix (sample {idx})')
    plt.xlabel('Key index'), plt.ylabel('Query index')
    plt.tight_layout();  plt.show()

# ---------------------------
# 测试过程
# ---------------------------

def test(args):
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备：", device)

    # 初始化模型并加载权重
    model = UNet(n_channels=2, n_classes=1, bilinear=True)
    translator = Generator(1,1)
    # translator.load_state_dict(torch.load('./models/299_netG_A2B.pth'))
    translator.load_state_dict(torch.load('./checkpoints/seg+cls\distill/3_G.pth'))

    # psa = PixelSelfAttention(in_channels=1, embed_dim=3).cuda()
    # psa_ = PixelSelfAttention(in_channels=1, proj=False).cuda()
    # psa.load_state_dict(torch.load('./checkpoints/Seg+cls/distill+atten/16_psa.pth'))

    model.to(device)
    translator.to(device)


    if not os.path.isfile(args.checkpoint):
        raise ValueError("模型权重文件不存在：{}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("加载模型权重：", args.checkpoint)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取测试图像列表
    image_list = sorted(os.listdir(args.test_image_dir))
    t1ce_image_list = sorted(os.listdir(args.test_t1ce_image_dir))

    if not image_list:
        print("测试图像目录为空：", args.test_image_dir)
        return


    for index, image_name in enumerate(image_list):
        image_path = os.path.join(args.test_image_dir, image_name)
        t1ce_image_path = os.path.join(args.test_t1ce_image_dir, t1ce_image_list[index])
        try:
            image = load_image(image_path)
            t1ce_image = load_image(t1ce_image_path)
        except Exception as e:
            print("读取图像失败：", image_path, e)
            continue
        # 增加 batch 维度
        image = image.unsqueeze(0).to(device)
        # t1ce_image = t1ce_image.unsqueeze(0).to(device)
        # t1ce_image = (t1ce_image - 0.5) * 2.0

        t1ce_image = translator((image - 0.5)*2.0)
        save_image((t1ce_image + 1.0)*0.5, './output/Seg+cls/trans_distill/' + image_name + '.png')

        with torch.no_grad():
            # show_full_matrix(psa_(F.interpolate(image, size=(64, 64), mode='bilinear')))
            output = model(torch.cat([image, (t1ce_image+1.0)*0.5], dim=1))
            # output = model(torch.cat([image, image], dim=1))
            # 由于使用 BCEWithLogitsLoss 训练，模型输出未经激活，需用 sigmoid 归一化
            output = torch.sigmoid(output)
            # 使用阈值 0.5 得到二值 mask
            mask = (output > 0.5).float()
            # 去除 batch 维度，并转为 numpy 数组
            mask_np = mask.squeeze().cpu().numpy()

        # 保存分割结果，文件名后缀添加 _mask
        # base, _ = os.path.splitext(image_name)
        # output_path = os.path.join(args.output_dir, base + "_mask.png")
        # save_mask(mask_np, output_path)
        # print("保存分割结果：", output_path)


# ---------------------------
# 主函数，解析命令行参数
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试二分类U-Net语义分割模型")

    parser.add_argument("--test_image_dir", type=str, default='./data/test/T1_4/', help="测试图像目录")
    # parser.add_argument("--test_image_dir", type=str, default='./data/classification/val/T1_0/', help="测试图像目录")

    parser.add_argument("--test_t1ce_image_dir", type=str, default='./data/test/T1ce_4/', help="测试图像目录")
    # parser.add_argument("--test_t1ce_image_dir", type=str, default='./data/classification/val/T1ce_0/', help="测试图像目录")

    parser.add_argument("--checkpoint", type=str, default='./checkpoints/seg+cls\distill/best_model_3.pth', help="训练好的模型权重文件路径")
    # parser.add_argument("--output_dir", type=str, default="./output/T1_4_concat_fake_Fusion", help="分割结果保存目录")
    parser.add_argument("--output_dir", type=str, default="./output/Seg+cls/T1_cls_distill", help="分割结果保存目录")
    args = parser.parse_args()

    test(args)
