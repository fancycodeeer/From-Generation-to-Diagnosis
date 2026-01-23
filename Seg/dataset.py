
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np
import os
import torchvision.transforms as transforms
import random

# ---------------------------
# 自定义数据集
# ---------------------------

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, t1ce_dir=None, transform=None):
        """
        image_dir: 图像存放目录
        mask_dir: 对应分割mask存放目录（假设mask为灰度图，值为0或255）
        transform: 数据预处理或数据增强
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.t1ce_dir = t1ce_dir

        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        if t1ce_dir is not None:
            self.t1ce_list = sorted(os.listdir(t1ce_dir))

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 读取图像和mask
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = cv2.imread(img_path)  # BGR格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 归一化处理
        image = image / 255.0
        mask = mask / 255.0  # mask值在0和1之间

        # 转换为Tensor，注意图片通道转换为 [C, H, W]
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        # mask增加通道维度
        mask = torch.from_numpy(np.expand_dims(mask, axis=0)).float()

        if self.transform:
            image = self.transform(image)

        if self.t1ce_dir is not None:
            t1ce_path = os.path.join(self.t1ce_dir, self.t1ce_list[idx])
            image_t1ce = cv2.imread(t1ce_path)
            image_t1ce = cv2.cvtColor(image_t1ce, cv2.COLOR_BGR2RGB)
            image_t1ce = image_t1ce / 255.0
            image_t1ce = torch.from_numpy(image_t1ce.transpose((2, 0, 1))).float()
            return image, mask, image_t1ce
        else:
            return image, mask


class ClassificationDataset(Dataset):
    def __init__(self, class_1_dir, class_2_dir, t1ce_1_dir, t1ce_2_dir, transform=None):
        """
        class_1_dir: 类别 1 图像存放目录
        class_2_dir: 类别 2 图像存放目录
        transform: 数据预处理或数据增强
        """
        self.class_1_dir = class_1_dir
        self.class_2_dir = class_2_dir
        self.t1ce_1_dir = t1ce_1_dir
        self.t1ce_2_dir = t1ce_2_dir

        self.class_1_images = sorted(os.listdir(class_1_dir))
        self.class_2_images = sorted(os.listdir(class_2_dir))
        self.t1ce_1_images = sorted(os.listdir(t1ce_1_dir))
        self.t1ce_2_images = sorted(os.listdir(t1ce_2_dir))

        self.transform = transform

    def __len__(self):
        # 每个批次从两类中选择一个样本，假设 batch_size = 2
        return max(len(self.class_1_images), len(self.class_2_images))

    def __getitem__(self, idx):
        # 随机选择类别 1 和 类别 2 图像的索引
        class_1_idx = random.randint(0, len(self.class_1_images) - 1)
        class_2_idx = random.randint(0, len(self.class_2_images) - 1)

        # 读取类别 1 图像
        class_1_img_path = os.path.join(self.class_1_dir, self.class_1_images[class_1_idx])
        class_1_image = cv2.imread(class_1_img_path)  # BGR格式
        class_1_image = cv2.cvtColor(class_1_image, cv2.COLOR_BGR2RGB)  # 转换为RGB
        class_1_image = ((class_1_image / 255.0) - 0.5) * 2.0  # 归一化
        class_1_image = torch.from_numpy(class_1_image.transpose((2, 0, 1))).float()

        t1ce_1_img_path = os.path.join(self.t1ce_1_dir, self.t1ce_1_images[class_1_idx])
        t1ce_1_image = cv2.imread(t1ce_1_img_path)  # BGR格式
        t1ce_1_image = cv2.cvtColor(t1ce_1_image, cv2.COLOR_BGR2RGB)  # 转换为RGB
        t1ce_1_image = ((t1ce_1_image / 255.0) - 0.5) * 2.0  # 归一化
        t1ce_1_image = torch.from_numpy(t1ce_1_image.transpose((2, 0, 1))).float()

        # 读取类别 2 图像
        class_2_img_path = os.path.join(self.class_2_dir, self.class_2_images[class_2_idx])
        class_2_image = cv2.imread(class_2_img_path)  # BGR格式
        class_2_image = cv2.cvtColor(class_2_image, cv2.COLOR_BGR2RGB)  # 转换为RGB
        class_2_image = ((class_2_image / 255.0) - 0.5) * 2.0  # 归一化
        class_2_image = torch.from_numpy(class_2_image.transpose((2, 0, 1))).float()

        t1ce_2_img_path = os.path.join(self.t1ce_2_dir, self.t1ce_2_images[class_2_idx])
        t1ce_2_image = cv2.imread(t1ce_2_img_path)  # BGR格式
        t1ce_2_image = cv2.cvtColor(t1ce_2_image, cv2.COLOR_BGR2RGB)  # 转换为RGB
        t1ce_2_image = ((t1ce_2_image / 255.0) - 0.5) * 2.0  # 归一化
        t1ce_2_image = torch.from_numpy(t1ce_2_image.transpose((2, 0, 1))).float()

        # 处理数据增强
        if self.transform:
            class_1_image = self.transform(class_1_image)
            class_2_image = self.transform(class_2_image)
            t1ce_1_image = self.transform(t1ce_1_image)
            t1ce_2_image = self.transform(t1ce_2_image)


        # 返回一个批次的两个图像和对应标签
        # 类别 1 标签是 0，类别 2 标签是 1
        return class_1_image, t1ce_1_image, class_2_image, t1ce_2_image
