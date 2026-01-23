
import torch
# import tensorflow as tf
import os
import logging
import math
import torch.fft as fft

# def restore_checkpoint(ckpt_dir, state, device):
#   if not tf.io.gfile.exists(ckpt_dir):
#     tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
#     logging.warning(f"No checkpoint found at {ckpt_dir}. "
#                     f"Returned the same state as input")
#     return state
#   else:
#     loaded_state = torch.load(ckpt_dir, map_location=device)
#     state['optimizer'].load_state_dict(loaded_state['optimizer'])
#     state['model'].load_state_dict(loaded_state['model'], strict=False)
#     state['ema'].load_state_dict(loaded_state['ema'])
#     state['step'] = loaded_state['step']
#     return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)



###############
# 傅里叶变换
###############
#----------------------------------------#
#   预处理训练图片
#----------------------------------------#
def preprocess_input(x):
    x = x / 255   # Out-of-place operation
    x = x - 0.5   # Out-of-place operation
    x = x / 0.5   # Out-of-place operation
    return x

def postprocess_output(x):
    x = x * 0.5
    x = x + 0.5
    x = x + 255
    return x


def make_distance_mask_(size):
    H, W = size[2], size[3]
    center_x, center_y = H / 2, W / 2

    x = torch.arange(H).float().cuda() - center_x
    y = torch.arange(W).float().cuda() - center_y

    xx, yy = torch.meshgrid(x, y, indexing="ij")
    distance = torch.sqrt(xx ** 2 + yy ** 2)

    return distance


def make_mask_(size, distance, threshold_h, threshold_l):

    mask_H = 1 - torch.exp(-(distance ** 2) / (2 * threshold_h ** 2))
    mask_L = torch.exp(-(distance ** 2) / (2 * threshold_l ** 2))

    mask_H = mask_H.unsqueeze(0).unsqueeze(0).repeat(size[0], size[1], 1, 1)
    mask_L = mask_L.unsqueeze(0).unsqueeze(0).repeat(size[0], size[1], 1, 1)

    return mask_H, mask_L


def Fourier_trans(img_tensor, mask_H, mask_L):
    input = postprocess_output(img_tensor)

    fourier_transform_0 = fft.rfft2(input)
    fourier_transform = fft.fftshift(fourier_transform_0)
    fourier_transform_L = mask_L * fourier_transform
    fourier_transform_H = mask_H * fourier_transform


    fourier_transform_L = fft.ifftshift(fourier_transform_L)
    fourier_transform_H = fft.ifftshift(fourier_transform_H)

    # Perform the inverse Fourier transform
    filtered_img_tensor_L = preprocess_input(torch.abs(fft.irfft2(fourier_transform_L)))
    filtered_img_tensor_H = preprocess_input(torch.abs(fft.irfft2(fourier_transform_H)))
    return filtered_img_tensor_H, filtered_img_tensor_L