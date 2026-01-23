import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import xml.etree.ElementTree as ET

# -------------------------------- Dataset --------------------------------
class CenterNetDataset(Dataset):
    """
    VOC-style dataset for CenterNet on single-channel MRI images.
    Parses 'tumor' objects, generates heatmap, size and offset targets.
    """
    def __init__(self, img_dir, ann_dir, transform=None, input_size=256, stride=1):
        super().__init__()
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.input_size = input_size  # input image size
        self.stride = stride          # downscale factor to feature map
        self.images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # --- Load and preprocess image ---
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # single-channel
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # shape [1, H, W]

        # --- Parse VOC XML annotation ---
        ann_path = os.path.join(self.ann_dir, os.path.splitext(img_name)[0] + '.xml')
        boxes, labels = [], []
        if os.path.exists(ann_path):
            tree = ET.parse(ann_path)
            for obj in tree.findall('object'):
                cls = obj.find('name').text.lower()
                if cls in ['tumor','foreground']:
                    bbox = obj.find('bndbox')
                    x1 = float(bbox.find('xmin').text)

                    y1 = float(bbox.find('ymin').text)
                    x2 = float(bbox.find('xmax').text)
                    y2 = float(bbox.find('ymax').text)
                    # scale coords to resized image
                    h0, w0 = img.shape
                    boxes.append([x1 * self.input_size / w0,
                                  y1 * self.input_size / h0,
                                  x2 * self.input_size / w0,
                                  y2 * self.input_size / h0])
                    labels.append(1)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # --- Generate CenterNet targets ---
        feat_size = self.input_size // self.stride
        heatmap = torch.zeros((1, feat_size, feat_size), dtype=torch.float32)
        wh       = torch.zeros((2, feat_size, feat_size), dtype=torch.float32)
        reg      = torch.zeros((2, feat_size, feat_size), dtype=torch.float32)
        reg_mask = torch.zeros((feat_size, feat_size), dtype=torch.uint8)
        indices  = []
        # For each ground-truth box, compute center cell and draw gaussian
        for box in boxes:
            x1, y1, x2, y2 = box / self.stride
            cx, cy = (x1+x2)/2, (y1+y2)/2
            w, h = x2-x1, y2-y1

            ix, iy = int(cx), int(cy)
            if 0 <= ix < feat_size and 0 <= iy < feat_size:
                # Compute gaussian radius and draw on heatmap
                radius = max(1, int(min(w,h))/2)
                sigma = radius / 3
                # if sigma <= 0:
                #     print(f"Invalid sigma: {sigma}")  # 打印无效的sigma
                #     continue
                yv = torch.arange(feat_size, dtype=torch.float32)
                xv = torch.arange(feat_size, dtype=torch.float32)
                yy, xx = torch.meshgrid(yv, xv)
                g = torch.exp(-((xx-ix)**2 + (yy-iy)**2) / (2*sigma**2))
                heatmap[0] = torch.max(heatmap[0], g)
                # Store width-height and offset at center
                wh[0, iy, ix] = w
                wh[1, iy, ix] = h
                reg[0, iy, ix] = cx - ix
                reg[1, iy, ix] = cy - iy
                reg_mask[iy, ix] = 1
                indices.append(iy * feat_size + ix)
        indices = torch.tensor(indices, dtype=torch.int64) if indices else torch.zeros((0,),dtype=torch.int64)

        targets = {'heatmap':heatmap, 'wh':wh, 'reg':reg,
                   'reg_mask':reg_mask, 'indices':indices,
                   'boxes':boxes, 'labels':labels}
        # print(f"Radius: {radius}, Sigma: {sigma}, cx: {cx}, cy: {cy}, ix: {ix}, iy: {iy}")
        return img_tensor, targets


def detection_collate_fn(batch):
    # batch 是一个 list，元素是 (img_tensor, targets_dict)
    imgs, targets = zip(*batch)
    # imgs: tuple of [1,H,W] → Tensor[B,1,H,W]
    imgs = torch.stack(imgs, dim=0)
    # targets: tuple of dict → list
    return imgs, list(targets)

class ClassifyDataset(Dataset):
    """
    VOC-style dataset for CenterNet on single-channel MRI images.
    Parses 'tumor' objects, generates heatmap, size and offset targets.
    """
    def __init__(self, img_dir, ann_dir, transform=None, input_size=256, stride=1):
        super().__init__()
        self.img_dir = img_dir
        self.ann_dir = ann_dir

        self.transform = transform
        self.input_size = input_size  # input image size
        self.stride = stride          # downscale factor to feature map
        self.images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # --- Load and preprocess image ---
        img_name = self.images[idx]
        feat_size = self.input_size // self.stride
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # single-channel
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # shape [1, H, W]

        if "_healthy" in img_name:
            heatmap = torch.zeros((1, feat_size, feat_size), dtype=torch.float32)
            category = torch.zeros((1), dtype=torch.float32)
        elif "ct_tumor" in img_name:
            heatmap = torch.zeros((1, feat_size, feat_size), dtype=torch.float32)
            category = torch.ones((1), dtype=torch.float32)
        else:
            category = torch.ones((1), dtype=torch.float32)

            # --- Parse VOC XML annotation ---
            ann_path = os.path.join(self.ann_dir, os.path.splitext(img_name)[0] + '.xml')
            boxes, labels = [], []
            if os.path.exists(ann_path):
                tree = ET.parse(ann_path)
                for obj in tree.findall('object'):
                    cls = obj.find('name').text.lower()
                    if cls in ['tumor','foreground']:
                        bbox = obj.find('bndbox')
                        x1 = float(bbox.find('xmin').text)

                        y1 = float(bbox.find('ymin').text)
                        x2 = float(bbox.find('xmax').text)
                        y2 = float(bbox.find('ymax').text)
                        # scale coords to resized image
                        h0, w0 = img.shape
                        boxes.append([x1 * self.input_size / w0,
                                      y1 * self.input_size / h0,
                                      x2 * self.input_size / w0,
                                      y2 * self.input_size / h0])
                        labels.append(1)
            boxes = torch.tensor(boxes, dtype=torch.float32)


            # --- Generate CenterNet targets ---
            heatmap = torch.zeros((1, feat_size, feat_size), dtype=torch.float32)
            # For each ground-truth box, compute center cell and draw gaussian
            for box in boxes:
                x1, y1, x2, y2 = box / self.stride
                cx, cy = (x1+x2)/2, (y1+y2)/2
                w, h = x2-x1, y2-y1

                ix, iy = int(cx), int(cy)
                if 0 <= ix < feat_size and 0 <= iy < feat_size:
                    # Compute gaussian radius and draw on heatmap
                    radius = max(1, int(min(w,h))/2)
                    sigma = radius / 3
                    # if sigma <= 0:
                    #     print(f"Invalid sigma: {sigma}")  # 打印无效的sigma
                    #     continue
                    yv = torch.arange(feat_size, dtype=torch.float32)
                    xv = torch.arange(feat_size, dtype=torch.float32)
                    yy, xx = torch.meshgrid(yv, xv)
                    g = torch.exp(-((xx-ix)**2 + (yy-iy)**2) / (2*sigma**2))
                    heatmap[0] = torch.max(heatmap[0], g)

        return img_tensor, heatmap, category



class ClassifyDatasetStroke(Dataset):
    def __init__(self, img_dir, transform=None, input_size=256, stride=1):
        super().__init__()
        self.img_dir = img_dir

        self.transform = transform
        self.input_size = input_size  # input image size
        self.stride = stride          # downscale factor to feature map
        self.images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # --- Load and preprocess image ---
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # single-channel
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # shape [1, H, W]

        if "_Normal" in img_name:
            category = torch.tensor([0.0], dtype=torch.long, requires_grad=False)
        elif "_Bleeding" in img_name:
            category = torch.tensor([1.0], dtype=torch.long, requires_grad=False)
        elif "_Ischemia" in img_name:
            category = torch.tensor([2.0], dtype=torch.long, requires_grad=False)
        else:
            raise ("Error!")

        return img_tensor, category


class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, input_size=256):
        super().__init__()
        self.img_dir = img_dir
        self.msk_dir = mask_dir

        self.transform = transform
        self.input_size = input_size  # input image size
        self.images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # --- Load and preprocess image ---
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # single-channel
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # shape [1, H, W]

        if "_Normal" in img_name:
            mask = torch.zeros_like(img_tensor)
        else:
            msk_path = os.path.join(self.msk_dir, img_name)
            mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.input_size, self.input_size))
            mask = mask.astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)  # shape [1, H, W]

        return img_tensor, mask