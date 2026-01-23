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
    translator.load_state_dict(torch.load('./checkpoints/label_distill_without_p2p/best_G.pth'))
    # CAN = CrossModalAttentionNetwork(threshold_minmax=[0.5, 4.5, 5, 25], num_heads=2)
    # CAN.load_state_dict(torch.load('./checkpoints/32_CAN.pth'))
    # AFN = AttentionFusionNet(in_channels=3)
    # AFN.load_state_dict(torch.load('./checkpoints/32_AFN.pth'))
    model.to(device)
    translator.to(device)
    # CAN.to(device)
    # AFN.to(device)

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

    # Frequency parameters
    distance_val = make_distance_mask_([1, 1, 240, int(240 / 2 + 1)])

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
        # save_image((t1ce_image + 1.0)*0.5, './output/trans/' + image_name + '.png')

        with torch.no_grad():
            output = model(torch.cat([image, (t1ce_image + 1.0)*0.5], dim=1))
            # output = model(torch.cat([image, image], dim=1))
            # 由于使用 BCEWithLogitsLoss 训练，模型输出未经激活，需用 sigmoid 归一化
            output = torch.sigmoid(output)
            # 使用阈值 0.5 得到二值 mask
            mask = (output > 0.5).float()
            # 去除 batch 维度，并转为 numpy 数组
            mask_np = mask.squeeze().cpu().numpy()

        # 保存分割结果，文件名后缀添加 _mask
        base, _ = os.path.splitext(image_name)
        output_path = os.path.join(args.output_dir, base + "_mask.png")
        save_mask(mask_np, output_path)
        print("保存分割结果：", output_path)


# ---------------------------
# 主函数，解析命令行参数
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试二分类U-Net语义分割模型")
    parser.add_argument("--test_image_dir", type=str, default='./data/test/T1_4/', help="测试图像目录")
    parser.add_argument("--test_t1ce_image_dir", type=str, default='./data/test/T1ce_4/', help="测试图像目录")
    parser.add_argument("--checkpoint", type=str, default='./checkpoints\SFD_new_version/best_model.pth', help="训练好的模型权重文件路径")
    # parser.add_argument("--output_dir", type=str, default="./output/T1_4_concat_fake_Fusion", help="分割结果保存目录")
    parser.add_argument("--output_dir", type=str, default="./output/SFD", help="分割结果保存目录")
    args = parser.parse_args()

    test(args)
