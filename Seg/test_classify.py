import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from dataset import ClassificationDataset
from models import ResNet18, ResNetClassifier, UNet
import argparse
import numpy as np
import torchvision.transforms as transforms
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 禁止cudnn的非确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭cudnn的优化，避免因硬件配置不同导致的不一致性

def test(args):
    set_seed(args.seed)  # 设置种子
    # 数据增强
    transform = transforms.Compose([transforms.Grayscale()])

    # 数据集 & 数据加载器
    val_dataset = ClassificationDataset(args.train_image_0_dir, args.train_image_1_dir, args.train_image_t1ce_0_dir, args.train_image_t1ce_1_dir, transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # 加载模型
    # model = moganet_tiny().cuda()
    model = ResNet18(num_classes=2, dim=2).cuda()
    # model = convnext_tiny().cuda()
    model.load_state_dict(torch.load(args.model_checkpoint_path, weights_only=True))  # 加载训练好的模型

    # 评估模式
    model.eval()

    # 用于计算混淆矩阵的变量
    all_labels = []
    all_preds = []

    running_val_loss = 0.0

    with torch.no_grad():
        for index, (t1_1_image, t1ce_1_image, t1_2_image, t1ce_2_image) in enumerate(val_loader):
            input_1 = torch.cat([t1_1_image, t1ce_1_image], dim=1).cuda()
            input_2 = torch.cat([t1_2_image, t1ce_2_image], dim=1).cuda()

            # input_1 = t1_1_image.cuda()
            # input_2 = t1_2_image.cuda()


            # 获取模型输出
            outputs = model(input_1)  # 输出形状 [B, 2]
            outputs_1 = model(input_2)  # 输出形状 [B, 2]
            print('output:',outputs, outputs_1)


            # 计算损失
            label_0_val = torch.tensor([0.0]).cuda()  # 类别0的标签
            label_1_val = torch.tensor([1.0]).cuda()  # 类别1的标签

            # 对每个输出应用 sigmoid
            probs = torch.softmax(outputs, dim=1)  # 每个样本属于类别1的概率
            probs_1 = torch.softmax(outputs_1, dim=1)
            print('softmax:',probs, probs_1)


            # 选择最大概率的类别
            predicted = (probs[:, 0] < 0.5).float()  # 如果类别1的概率大于0.5，预测为类别1，否则为类别0
            predicted_1 = (probs_1[:, 0] < 0.5).float()  # 对第二个输出做同样的处理
            print(predicted, predicted_1)


            # 收集所有标签和预测值
            all_labels.extend(label_0_val.cpu().numpy())  # 真实标签
            all_preds.extend(predicted.cpu().numpy())  # 预测标签
            all_labels.extend(label_1_val.cpu().numpy())  # 真实标签
            all_preds.extend(predicted_1.cpu().numpy())  # 预测标签

    print(all_labels, all_preds)
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = conf_matrix.ravel()  # 解包混淆矩阵

    # 输出混淆矩阵和指标
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Negatives (TN): {TN}")

    # 计算其他评价指标
    accuracy = (TP + TN) / (TP + FP + FN + TN)  # 准确率
    precision = TP / (TP + FP)  # 精确率
    recall = TP / (TP + FN)  # 召回率
    f1 = 2 * (precision * recall) / (precision + recall)  # F1-Score

    # 输出评价指标
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="二分类模型验证脚本")
    parser.add_argument("--train_image_0_dir", type=str, default='./data/classification/val/T1_0')
    parser.add_argument("--train_image_t1ce_0_dir", type=str, default='./data/classification/val/T1ce_0')
    parser.add_argument("--train_image_1_dir", type=str, default='./data/classification/val/T1_4')
    parser.add_argument("--train_image_t1ce_1_dir", type=str, default='./data/classification/val/T1ce_4')
    parser.add_argument("--model_checkpoint_path", type=str, default='./checkpoints/classify_mix/best_model.pth', help="训练好的模型路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    test(args)
