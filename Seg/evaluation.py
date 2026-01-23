import os
import numpy as np
import torch
import cv2
from sklearn.metrics import confusion_matrix

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
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            # 归一化到 [0, 1] 之间
            pred = (pred / 255.0) > threshold  # 使用阈值进行二值化
            gt = gt / 255.0  # 将真实标签也转化为 [0, 1] 之间

            # 计算各项指标
            dice = calculate_dice_score(pred, gt)
            iou = calculate_iou(pred, gt)
            accuracy = calculate_accuracy(pred, gt)
            precision = calculate_precision(pred, gt)
            recall = calculate_recall(pred, gt)

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

if __name__ == "__main__":
    # pred_dir = "./output/T1_4_concat_fake_Fusion"  # 预测结果目录
    pred_dir = "./output/fddt4seg/cyc_sfd/"
    gt_dir = "./data/test/seg_4"  # 真实标签目录
    evaluate(pred_dir, gt_dir)
