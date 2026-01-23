import random
from torch.autograd import Variable
import torch
import torch.nn.init as init
import math
import torch.fft as fft

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


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

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

    xx, yy = torch.meshgrid(x, y)
    distance = torch.sqrt(xx ** 2 + yy ** 2)

    return distance.unsqueeze(0).unsqueeze(0).repeat(size[0], size[1], 1, 1)

def make_mask_(distance, threshold_h, threshold_l):  #  torch.Size([2, 1, 1 ,1])

    mask_H = 1 - torch.exp(-(distance ** 2) / (2 * threshold_h.view(-1, 1, 1, 1) ** 2))
    mask_L = torch.exp(-(distance ** 2) / (2 * threshold_l.view(-1, 1, 1, 1) ** 2))

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
    return [filtered_img_tensor_H, filtered_img_tensor_L]

