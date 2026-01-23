"""
纯 Python 版本的 upfirdn2d.py，不依赖 C++/CUDA 扩展。
该文件实现了 upfirdn2d 操作，用于先上采样、填充、滤波，再下采样。
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import abc


def _setup_kernel(k):
	k = np.asarray(k, dtype=np.float32)
	if k.ndim == 1:
		k = np.outer(k, k)
	k /= np.sum(k)
	assert k.ndim == 2 and k.shape[0] == k.shape[1]
	return k


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
	"""
    纯 Python 实现的 upfirdn2d 操作。

    Args:
        input (Tensor): 输入张量，形状 (N, C, H, W)。
        kernel (Tensor or ndarray): 二维滤波核，形状 (K, K)。
        up_x, up_y (int): 横向和纵向上采样因子。
        down_x, down_y (int): 横向和纵向下采样因子。
        pad_x0, pad_x1, pad_y0, pad_y1 (int): 填充大小。

    Returns:
        Tensor: 输出张量，形状根据上采样、填充、卷积和下采样计算得到。
    """
	_, channel, in_h, in_w = input.shape
	input = input.reshape(-1, in_h, in_w, 1)
	# 这里 minor 总为 1（因为 input 最后一维为 1）
	minor = 1
	kernel_h, kernel_w = kernel.shape if torch.is_tensor(kernel) else (kernel.shape[0], kernel.shape[1])

	# 将输入插入空维后上采样
	out = input.view(-1, in_h, 1, in_w, 1, minor)
	out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
	out = out.view(-1, in_h * up_y, in_w * up_x, minor)

	# 对上采样结果进行填充
	out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
	out = out[:, max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0),
	      max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0), :]

	# 变换为 (N, C, H, W) 形式并做卷积（使用翻转后的滤波核）
	out = out.permute(0, 3, 1, 2)
	out = out.reshape(-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1)
	# 若 kernel 为 ndarray，则转换为 tensor
	if not torch.is_tensor(kernel):
		kernel = torch.tensor(kernel, dtype=input.dtype, device=input.device)
	w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
	out = F.conv2d(out, w)
	out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
	                  in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
	out = out.permute(0, 2, 3, 1)
	out = out[:, ::down_y, ::down_x, :]

	out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
	out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
	return out.view(-1, channel, out_h, out_w)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
	"""
    upfirdn2d 函数：对输入进行上采样、填充、滤波、下采样。

    Args:
        input (Tensor): 输入张量，形状 (N, C, H, W)。
        kernel (Tensor or ndarray): 二维滤波核。
        up (int or tuple): 上采样因子（若为整数，则横纵方向相同）。
        down (int or tuple): 下采样因子（若为整数，则横纵方向相同）。
        pad (int or tuple): 填充（若为单个整数，则各边相同；若为二元组，则分别用于左右或上下）。

    Returns:
        Tensor: 输出张量。
    """
	if not isinstance(up, (tuple, list)):
		up = (up, up)
	if not isinstance(down, (tuple, list)):
		down = (down, down)
	if isinstance(pad, int):
		pad = (pad, pad, pad, pad)
	elif isinstance(pad, (tuple, list)) and len(pad) == 2:
		pad = (pad[0], pad[1], pad[0], pad[1])
	up_x, up_y = up
	down_x, down_y = down
	pad_x0, pad_x1, pad_y0, pad_y1 = pad
	return upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1)


def upfirdn2d_ada(input, kernel, up=1, down=1, pad=(0, 0)):
	"""
    兼容性版本，与 upfirdn2d 类似，不过允许 up/down/pad 为可迭代对象。
    """
	if not isinstance(up, abc.Iterable):
		up = (up, up)
	if not isinstance(down, abc.Iterable):
		down = (down, down)
	if len(pad) == 2:
		pad = (pad[0], pad[1], pad[0], pad[1])
	return upfirdn2d_native(input, kernel, up[0], up[1], down[0], down[1], pad[0], pad[1], pad[2], pad[3])
