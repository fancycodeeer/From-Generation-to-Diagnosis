#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import argparse
from models import CenterNet
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from models_t import Generator

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
    image = cv2.resize(image, (256, 256))
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
    model = CenterNet(num_classes=1, n_channels=2)
    model.to(device)
    if not os.path.isfile(args.checkpoint):
        raise ValueError("模型权重文件不存在：{}".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print("加载模型权重：", args.checkpoint)

    translator = Generator().to(device)
    translator.load_state_dict(torch.load(args.translator_checkpoint))
    translator.eval()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取测试图像列表
    image_list = sorted(os.listdir(args.test_image_dir))
    if not image_list:
        print("测试图像目录为空：", args.test_image_dir)
        return

    for image_name in image_list:
        image_path = os.path.join(args.test_image_dir, image_name)
        try:
            image = load_image(image_path)
        except Exception as e:
            print("读取图像失败：", image_path, e)
            continue
        # 增加 batch 维度
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            fake_mr = translator((image - 0.5) * 2)
            fake_mr = (fake_mr + 1) * 0.5
            output = model(torch.cat([image, fake_mr], dim=1))
            # output = model(image)
            # 由于使用 BCEWithLogitsLoss 训练，模型输出未经激活，需用 sigmoid 归一化
            # output = torch.sigmoid(output)
            # 使用阈值 0.5 得到二值 mask
            mask = (output > 0.5).float()
            # 去除 batch 维度，并转为 numpy 数组
            mask_np = mask.squeeze().cpu().numpy()

        # 保存分割结果，文件名后缀添加 _mask
        base, _ = os.path.splitext(image_name)
        output_path = os.path.join(args.output_dir, base + "_mask.png")
        save_mask(mask_np, output_path)
        print("保存分割结果：", output_path)

def calculate_dice_score(pred, target):
    """
    计算Dice系数（Jaccard指数的变体）
    pred: 预测的二值图像（numpy数组或tensor，值为 0 或 1）
    target: 真实标签的二值图像（numpy数组或tensor，值为 0 或 1）
    """
    intersection = np.sum(pred * target)
    return 2. * intersection / (np.sum(pred) + np.sum(target) + 1e-5)

def calculate_iou(pred, target):
    """
    计算IoU（Intersection over Union）
    """
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return intersection / (union + 1e-5)

def calculate_accuracy(pred, target):
    """
    计算精度（Accuracy）
    """
    correct = np.sum(pred == target)
    total = target.size
    return correct / total

def calculate_precision(pred, target):
    """
    计算精准率（Precision）
    """
    tn, fp, fn, tp = confusion_matrix(target.flatten(), pred.flatten()).ravel()
    return tp / (tp + fp + 1e-5)

def calculate_recall(pred, target):
    """
    计算召回率（Recall）
    """
    tn, fp, fn, tp = confusion_matrix(target.flatten(), pred.flatten()).ravel()
    return tp / (tp + fn + 1e-5)

def evaluate(pred_dir, gt_dir, threshold=0.5):
    """
    计算指定目录下所有图像的Dice, IoU, 精度, 精准率和召回率。
    pred_dir: 预测结果目录，包含二值图像（.png 或 .jpg 等）
    gt_dir: 真实标签目录，包含二值图像
    threshold: 预测结果的阈值，默认为 0.5
    """
    dice_scores = []
    iou_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))

    for pred_file, gt_file in zip(pred_files, gt_files):
       if pred_file.endswith('.png') and gt_file.endswith('.png'):
            pred_path = os.path.join(pred_dir, pred_file)
            gt_path = os.path.join(gt_dir, gt_file)

            # 读取图像，假设图像是二值化的
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.resize(pred, (256, 256))
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (256, 256))

            # 归一化到 [0, 1] 之间
            pred = (pred / 255.0) > threshold  # 使用阈值进行二值化
            gt = gt / 255.0  # 将真实标签也转化为 [0, 1] 之间

            # 计算各项指标
            dice = calculate_dice_score(pred, gt)
            iou = calculate_iou(pred, gt)
            accuracy = calculate_accuracy(pred, gt)
            precision = calculate_precision(pred.astype(int), gt.astype(int))
            recall = calculate_recall(pred.astype(int), gt.astype(int))

            dice_scores.append(dice)
            iou_scores.append(iou)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)

    # 打印平均值
    print(f"Dice Score: {np.mean(dice_scores):.4f}")
    print(f"IoU Score: {np.mean(iou_scores):.4f}")
    print(f"Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"Precision: {np.mean(precision_scores):.4f}")
    print(f"Recall: {np.mean(recall_scores):.4f}")
# ---------------------------
# 主函数，解析命令行参数
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试二分类U-Net语义分割模型")
    parser.add_argument("--test_image_dir", type=str, default='./data4seg/val/img_/', help="测试图像目录")
    parser.add_argument("--checkpoint", type=str, default='./logs/seg_dist_/best_dice.pth', help="训练好的模型权重文件路径")
    parser.add_argument("--output_dir", type=str, default="./output/seg/ischemia_dist", help="分割结果保存目录")
    parser.add_argument("--translator_checkpoint", type=str, default='./logs/seg_dist_/90_g.pth', help="分割结果保存目录")
    args = parser.parse_args()

    test(args)
    pred_dir = "./output/seg/ischemia_dist"
    gt_dir = "./data4seg/val/maks_/"  # 真实标签目录
    evaluate(pred_dir, gt_dir)
