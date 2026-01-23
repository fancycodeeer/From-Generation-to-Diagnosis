import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
# from visdom import Visdom
import numpy as np
import torch.fft as fft
# from DCT import dct2, idct2
import math
import torchvision.transforms as transforms
import torch.nn as nn

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        # self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}


    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i+1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))

        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # Draw images
        # for image_name, tensor in images.items():
        #     if image_name not in self.image_windows:
        #         self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title':image_name})
        #     else:
        #         self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name], opts={'title':image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                # if loss_name not in self.loss_windows:
                #     self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]),
                #                                                     opts={'xlabel': 'epochs', 'ylabel': loss_name, 'title': loss_name})
                # else:
                #     self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1

        

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def norm(input):
    # for i in range(240):
    #     for j in range(240):
    #         if int(input[0, 0, i, j]) <= 0:
    #             input[:, :, i, j] = 0
    max = float(torch.max(input))
    min = float(torch.min(input))
    out = (input - min) * (255.0/(max - min))
    return out


def Fourier_trans(img_tensor, mask_H, mask_L, exchange=False):
    input = (img_tensor * 0.5 + 0.5) * 255.0

    fourier_transform_0 = fft.rfft2(input)
    fourier_transform = fft.fftshift(fourier_transform_0)
    fourier_transform_L = mask_L * fourier_transform
    fourier_transform_H = mask_H * fourier_transform

    if exchange:
        return [fourier_transform_H, fourier_transform_L]
    else:
        fourier_transform_L = fft.ifftshift(fourier_transform_L)
        fourier_transform_H = fft.ifftshift(fourier_transform_H)

        # Perform the inverse Fourier transform
        filtered_img_tensor_L = torch.abs(fft.irfft2(fourier_transform_L))
        filtered_img_tensor_H = torch.abs(fft.irfft2(fourier_transform_H))
        return [filtered_img_tensor_H, filtered_img_tensor_L]


    # B, C, W, H = fourier_transform.shape[0], fourier_transform.shape[1], fourier_transform.shape[2], fourier_transform.shape[3]
    # mask_L = torch.zeros(B, C, W, H).cuda()
    # mask_H = torch.zeros(B, C, W, H).cuda()

    # c_w = W / 2; c_h = H / 2
    # # 高斯滤波器
    # distance = torch.zeros(B, C, W, H).cuda()
    # for w in range(W):
    #     for h in range(H):
    #         distance[:, :, w, h] = math.sqrt((pow((w - c_w), 2)) + pow((h - c_h), 2))
    #         mask_H[:, :, w, h] = 1 - torch.exp(-(torch.pow(distance[:,:, w, h], 2)) / (2 * D * D))  # 高通
            # mask_L[:, :, w, h] = torch.exp(-(torch.pow(distance[:, :, w, h], 2)) / (2 * D * D))  # 低通
    #
    # for w in range(W):
    #     for h in range(H):
    #             # if math.sqrt((pow((w-c_w),2)) + pow((h-c_h),2)) < D:
    #             #     mask_L[:, :, w, h] = 1
    #             if math.sqrt((pow((w - c_w), 2)) + pow((h - c_h), 2)) >= D:
    #                 mask_H[:, :, w, h] = 1



def init_transform(input):
    output = []
    for i in range(len(input)):
        output.append(transforms.Normalize((0.5), (0.5))(input[i] / 255.0))
    return output


def make_mask(size, D):
    mask_H = torch.zeros(size[0],size[1],size[2],size[3]).cuda()
    mask_L = torch.zeros(size[0], size[1], size[2], size[3]).cuda()
    c_w = size[2] / 2
    c_h = size[3] / 2
    # 高斯滤波器
    distance = torch.zeros(size[0],size[1],size[2],size[3]).cuda()
    for w in range(size[2]):
        for h in range(size[3]):
            distance[:, :, w, h] = math.sqrt((pow((w - c_w), 2)) + pow((h - c_h), 2))
            mask_H[:, :, w, h] = 1 - torch.exp(-(torch.pow(distance[:, :, w, h], 2)) / (2 * D * D)) # 高通
            mask_L[:, :, w, h] = torch.exp(-(torch.pow(distance[:, :, w, h], 2)) / (2 * D * D)) # 低通

    return mask_H, mask_L

def exchange_ifft(img1, img2):
    img1_H = img1[0]
    img1_L = img1[1]
    img2_H = img2[0]
    img2_L = img2[1]

    exchange1 = fft.irfft2(fft.ifftshift(img1_H + img2_L))
    exchange2 = fft.irfft2(fft.ifftshift(img2_H + img1_L))

    return [exchange1, exchange2]


def make_mask_dct(size, D):
    mask_H = torch.zeros(size[0],size[1],size[2],size[3]).cuda()
    mask_L = torch.zeros(size[0], size[1], size[2], size[3]).cuda()
    c_w = 0  # size[2] / 2
    c_h = 0  # size[3] / 2
    # 高斯滤波器
    distance = torch.zeros(size[0],size[1],size[2],size[3])
    for w in range(size[2]):
        for h in range(size[3]):
            distance[:, :, w, h] = math.sqrt((pow((w - c_w), 2)) + pow((h - c_h), 2))
            mask_H[:, :, w, h] = 1 - torch.exp(-(torch.pow(distance[:, :, w, h], 2)) / (2 * D * D)) # 高通
            mask_L[:, :, w, h] = torch.exp(-(torch.pow(distance[:, :, w, h], 2)) / (2 * D * D)) # 低通

    return mask_H, mask_L

# def DCT_trans(img_tensor, mask_H, mask_L, dct_matrix):
#     input = (img_tensor * 0.5 + 0.5) * 255.0
#
#     dct_coeffs = dct2(input, dct_matrix)
#     h_dct_coeffs = dct_coeffs * mask_H
#     l_dct_coeffs = dct_coeffs * mask_L
#     h_inverse_dct_coeffs = idct2(h_dct_coeffs, dct_matrix)
#     l_inverse_dct_coeffs = idct2(l_dct_coeffs, dct_matrix)
#     return [h_inverse_dct_coeffs, l_inverse_dct_coeffs]


def make_dct_matrix(batch_size, channels, height, width, dtype, device):
    dct_matrix = torch.zeros(batch_size, channels, height, width, dtype=dtype, device=device)
    pi_tensor = torch.tensor(math.pi, dtype=dtype, device=device)

    for i in range(height):
        for j in range(width):
            if i == 0:
                a = torch.tensor(math.sqrt(1.0/height), dtype=dtype, device=device)
            else:
                a = torch.tensor(math.sqrt(2.0/height), dtype=dtype, device=device)

            dct_matrix[:, :, i, j] = a * torch.cos(pi_tensor * (j + 0.5) * i / height)

    return dct_matrix


class AdaptiveFrequencyMask(nn.Module):
    def __init__(self, min_D=1, max_D=60):
        super(AdaptiveFrequencyMask, self).__init__()

        self.min_D = min_D
        self.max_D = max_D

        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            # 输入: 幅度谱和相位谱 (2通道)
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 空间压缩，降低计算量
            nn.AvgPool2d(2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 全局特征
        self.global_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def get_frequency_features(self, x):
        # 计算傅里叶变换
        fft = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft)

        # 提取幅度谱和相位谱
        magnitude = torch.log(torch.abs(fft_shift) + 1)  # 取对数压缩范围
        phase = torch.angle(fft_shift)

        # 拼接频域特征 [B, 2, H, W]
        freq_features = torch.cat([magnitude, phase], dim=1)
        return freq_features

    def predict_D(self, x):
        # 提取频域特征
        freq_features = self.get_frequency_features(x)

        # 特征提取
        features = self.feature_extractor(freq_features)

        # 预测D值
        normalized_D = self.global_features(features)
        D = self.min_D + (self.max_D - self.min_D) * normalized_D
        return D

    def make_mask(self, size, D, device):
        """
        Args:
            size: 输出掩码的尺寸 [B, C, H, W/2+1]
            D: 截断阈值 [B]
        Returns:
            mask_H: 高通滤波掩码
            mask_L: 低通滤波掩码
        """
        # 计算中心点
        c_h = size[2] / 2
        c_w = (size[3] - 1)

        # 生成网格坐标
        h_indices = torch.arange(size[2], device=device)
        w_indices = torch.arange(size[3], device=device)
        w_grid, h_grid = torch.meshgrid(w_indices, h_indices)

        # 计算距离矩阵
        distance = torch.sqrt((w_grid - c_w).pow(2) + (h_grid - c_h).pow(2))
        distance = distance.T.unsqueeze(0).unsqueeze(0).expand(size[0], size[1], -1, -1)

        # 调整D的维度[B] -> [B, 1, 1, 1]
        D = D.view(-1, 1, 1, 1)

        # 计算高斯因子
        gaussian_factor = torch.exp(-distance.pow(2) / (2 * D.pow(2)))

        # 生成高通和低通掩码
        mask_L = gaussian_factor
        mask_H = 1 - gaussian_factor

        return mask_H, mask_L

    def forward(self, x, size):
        # 预测D值
        D = self.predict_D(x)

        # 生成掩码
        mask_H, mask_L = self.make_mask(size, D, x.device)

        return mask_H, mask_L, D