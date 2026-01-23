#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute average PSNR & SSIM between two folders of images,
paired strictly by filename order (sorted). Supports RGB/Gray, alpha剥离，
并可选将B目录重采样到A图像尺寸。

Usage:
  python pairwise_psnr_ssim.py /path/to/A /path/to/B \
      [--exts png jpg jpeg bmp tif tiff webp] \
      [--recursive] \
      [--resize-to A|B|none] \
      [--mode auto|gray|rgb|y] \
      [--per-image] \
      [--csv out.csv]

Author: ChatGPT (for Wang)
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import sys

import numpy as np
from PIL import Image

# skimage 兼容导入
try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

def list_images(d: Path, exts: List[str], recursive: bool) -> List[Path]:
    exts = tuple("." + e.lower().lstrip(".") for e in exts)
    if recursive:
        files = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name.lower())
    return files

def pil_load_rgb_or_gray(path: Path) -> np.ndarray:
    """Load image via PIL. Return np.ndarray.
    - If image has alpha, drop it.
    - Keep either HxW (gray) or HxWx3 (RGB)."""
    im = Image.open(path)
    # 统一到 RGB 或 L，尽量不改变通道数，先判断原始模式
    if im.mode in ("RGBA", "LA"):
        # 去掉alpha
        base = "RGB" if im.mode == "RGBA" else "L"
        im = im.convert(base)
    elif im.mode == "P":
        # 调色板转RGB
        im = im.convert("RGB")
    elif im.mode not in ("RGB", "L"):
        # 其它模式统一处理
        im = im.convert("RGB")
    arr = np.array(im)
    return arr

def maybe_resize(arr: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    H, W = size_hw
    if arr.ndim == 2:
        img = Image.fromarray(arr)
        return np.array(img.resize((W, H), Image.BILINEAR))
    elif arr.ndim == 3:
        img = Image.fromarray(arr)
        return np.array(img.resize((W, H), Image.BILINEAR))
    else:
        raise ValueError("Unexpected array ndim: {}".format(arr.ndim))

def ensure_mode(arrA: np.ndarray, arrB: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    根据mode决定是灰度、RGB还是亮度Y通道：
    - auto: 若两者都是3通道则RGB，否则灰度
    - gray: 转灰度 (H,W)
    - rgb:  强制RGB (H,W,3)
    - y:    从RGB转换到Y通道 (H,W)
    返回 (A,B,is_color) 其中 is_color=True 表示SSIM按多通道处理
    """
    def to_gray(a):
        if a.ndim == 2:
            return a
        # RGB -> Gray (ITU-R BT.601)
        return np.dot(a[..., :3], [0.299, 0.587, 0.114]).astype(a.dtype)

    def to_rgb(a):
        if a.ndim == 2:
            return np.stack([a, a, a], axis=-1)
        elif a.ndim == 3 and a.shape[-1] == 3:
            return a
        else:
            raise ValueError("Unsupported array shape for RGB: {}".format(a.shape))

    def to_y(a):
        # 转到亮度通道Y
        if a.ndim == 2:
            return a
        rgb = a[..., :3].astype(np.float64)
        y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return y.astype(a.dtype)

    if mode == "gray":
        return to_gray(arrA), to_gray(arrB), False
    if mode == "rgb":
        return to_rgb(arrA), to_rgb(arrB), True
    if mode == "y":
        return to_y(arrA), to_y(arrB), False

    # auto
    a3 = (arrA.ndim == 3 and arrA.shape[-1] == 3)
    b3 = (arrB.ndim == 3 and arrB.shape[-1] == 3)
    if a3 and b3:
        return to_rgb(arrA), to_rgb(arrB), True
    else:
        return to_gray(arrA), to_gray(arrB), False

def data_range_from_dtype(orig_dtype) -> float:
    # 建议用dtype动态范围，而不是图像本身min/max
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        return float(info.max - info.min)
    # 浮点常见为[0,1]
    return 1.0

def compute_psnr(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    if _HAS_SKIMAGE:
        return float(sk_psnr(a, b, data_range=data_range))
    # 退化到手写实现
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(data_range) - 10.0 * np.log10(mse)

def compute_ssim(a: np.ndarray, b: np.ndarray, data_range: float, is_color: bool) -> float:
    if not _HAS_SKIMAGE:
        raise RuntimeError("需要 scikit-image 才能计算SSIM，请先安装：pip install scikit-image")
    # 新旧API兼容：channel_axis vs multichannel
    try:
        if is_color:
            return float(sk_ssim(a, b, channel_axis=-1, data_range=data_range))
        else:
            return float(sk_ssim(a, b, data_range=data_range))
    except TypeError:
        # 旧版
        if is_color:
            return float(sk_ssim(a, b, multichannel=True, data_range=data_range))
        else:
            return float(sk_ssim(a, b, multichannel=False, data_range=data_range))

def main():
    parser = argparse.ArgumentParser(description="Pairwise PSNR/SSIM by filename order.")
    parser.add_argument("--dirA", default='E:\BraTS2024\外部性检验/Flair',type=Path,help="目录A（基准）")
    parser.add_argument("--dirB", default='E:\BraTS2024\外部性检验/T22Flair/unit+fddt/280/B',type=Path, help="目录B")
    parser.add_argument("--exts", nargs="+",default=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],help="识别的图片扩展名（大小写不敏感）")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument("--resize-to", choices=["A", "B", "none"], default="A",help="尺寸不一致时的处理：将B重采样到A尺寸（选A），将A重采样到B尺寸（选B），或报错（none）")
    parser.add_argument("--mode", choices=["auto", "gray", "rgb", "y"], default="auto",help="比较模式：auto(默认)、gray(灰度)、rgb(三通道)、y(亮度通道)")
    parser.add_argument("--per-image", action="store_true", help="逐图打印PSNR/SSIM")
    parser.add_argument("--csv", type=Path, default=None, help="将逐图结果保存到CSV")
    args = parser.parse_args()

    if not args.dirA.is_dir() or not args.dirB.is_dir():
        print("错误：请提供两个存在的目录。", file=sys.stderr)
        sys.exit(1)

    filesA = list_images(args.dirA, args.exts, args.recursive)
    filesB = list_images(args.dirB, args.exts, args.recursive)

    if len(filesA) == 0 or len(filesB) == 0:
        print("错误：至少一个目录未找到匹配的图片文件。", file=sys.stderr)
        sys.exit(1)

    n_pairs = min(len(filesA), len(filesB))
    if len(filesA) != len(filesB):
        print(f"警告：文件数不一致，按排序仅比较前 {n_pairs} 对。A={len(filesA)}, B={len(filesB)}", file=sys.stderr)

    psnrs = []
    ssims = []
    rows = []

    for i in range(n_pairs):
        pa = filesA[i]
        pb = filesB[i]
        try:
            a = pil_load_rgb_or_gray(pa)
            b = pil_load_rgb_or_gray(pb)
        except Exception as e:
            print(f"[跳过] 读取失败: {pa.name} vs {pb.name} -> {e}", file=sys.stderr)
            continue

        # 尺寸处理
        if a.shape[:2] != b.shape[:2]:
            if args.resize_to == "A":
                b = maybe_resize(b, a.shape[:2])
            elif args.resize_to == "B":
                a = maybe_resize(a, b.shape[:2])
            else:
                print(f"[跳过] 尺寸不一致且未指定重采样: {pa.name}({a.shape}) vs {pb.name}({b.shape})", file=sys.stderr)
                continue

        # 模式统一
        A_mode_orig_dtype = a.dtype
        B_mode_orig_dtype = b.dtype
        a, b, is_color = ensure_mode(a, b, args.mode)

        # 数据范围基于dtype
        # 两者可能不同dtype，取较大的动态范围以避免偏置
        drA = data_range_from_dtype(A_mode_orig_dtype)
        drB = data_range_from_dtype(B_mode_orig_dtype)
        data_range = max(drA, drB)

        try:
            p = compute_psnr(a, b, data_range=data_range)
            s = compute_ssim(a, b, data_range=data_range, is_color=is_color)
        except Exception as e:
            print(f"[跳过] 指标计算失败: {pa.name} vs {pb.name} -> {e}", file=sys.stderr)
            continue

        psnrs.append(p)
        ssims.append(s)
        if args.per_image:
            print(f"{i+1:4d}. {pa.name:>32s}  ||  {pb.name:>32s}  |  PSNR={p:.4f}  SSIM={s:.6f}")
        if args.csv is not None:
            rows.append((pa.name, pb.name, p, s))

    if len(psnrs) == 0:
        print("没有成功比较的样本。", file=sys.stderr)
        sys.exit(2)

    # 处理无穷大PSNR的平均
    psnrs_arr = np.array(psnrs, dtype=float)
    finite_mask = np.isfinite(psnrs_arr)
    if finite_mask.any():
        avg_psnr = float(psnrs_arr[finite_mask].mean())
        inf_count = int((~finite_mask).sum())
    else:
        avg_psnr = float("inf")
        inf_count = len(psnrs)

    avg_ssim = float(np.mean(ssims))

    print("\n=== Summary ===")
    print(f"Pairs compared          : {len(psnrs)}")
    print(f"Average SSIM            : {avg_ssim:.6f}")
    print(f"Average PSNR (dB)       : {avg_psnr:.6f}" if np.isfinite(avg_psnr) else "Average PSNR (dB)       : inf")
    if inf_count > 0:
        print(f"  (Note: {inf_count} pairs identical -> PSNR=inf, 已从均值中剔除计算)")
    print(f"Mode                    : {args.mode}  |  Resize: {args.resize_to}")

    # if args.csv is not None and len(rows) > 0:
    #     try:
    #         import csv
    #         with open(args.csv, "w", newline="", encoding="utf-8") as f:
    #             w = csv.writer(f)
    #             w.writerow(["index", "A_filename", "B_filename", "PSNR(dB)", "SSIM"])
    #             for idx, (na, nb, p, s) in enumerate(rows, 1):
    #                 w.writerow([idx, na, nb, f"{p:.6f}", f"{s:.6f}"])
    #         print(f"Per-image metrics saved to: {args.csv}")
    #     except Exception as e:
    #         print(f"CSV 保存失败：{e}", file=sys.stderr)

if __name__ == "__main__":
    main()
