#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models_FSeSim import Generator, Discriminator, FSeSimLoss
from utils_b import Fourier_trans, init_transform, make_mask, exchange_ifft
from utils_b import ReplayBuffer
from utils_b import LambdaLR
from utils_b import Logger
from utils_b import weights_init_normal
from datasets_b import ImageDataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import torch
import numpy as np
import random


def set_seed(seed):
	random.seed(seed)  # Python 随机数生成器
	np.random.seed(seed)  # NumPy 随机数生成器
	torch.manual_seed(seed)  # CPU 上的 PyTorch 随机数生成器
	torch.cuda.manual_seed(seed)  # GPU 上的 PyTorch 随机数生成器
	torch.cuda.manual_seed_all(seed)  # 多 GPU
	torch.backends.cudnn.deterministic = True  # 确保卷积操作可重复
	torch.backends.cudnn.benchmark = False  # 禁用动态优化


# 调用设置
set_seed(99)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./datasets/CHAOS/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_false', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--multiGPU', type=bool, default=True, help='Use multiGPU train')
parser.add_argument('--local_rank', type=int, default=0)
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and opt and opt.multiGPU:
	local_rank = opt.local_rank
	torch.cuda.set_device(local_rank)
	torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=local_rank)
	torch.cuda.set_device(local_rank)

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

netG_A2B.cuda()
netG_B2A.cuda()
netD_A.cuda()
netD_B.cuda()


netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

if opt.multiGPU:
	netG_A2B = torch.nn.parallel.DistributedDataParallel(netG_A2B, device_ids=[local_rank], output_device=local_rank)
	netG_B2A = torch.nn.parallel.DistributedDataParallel(netG_B2A, device_ids=[local_rank], output_device=local_rank)
	netD_A = torch.nn.parallel.DistributedDataParallel(netD_A, device_ids=[local_rank], output_device=local_rank)
	netD_B = torch.nn.parallel.DistributedDataParallel(netD_B, device_ids=[local_rank], output_device=local_rank)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
FSeSimLoss = FSeSimLoss().cuda()


# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.Grayscale(),
               transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5), (0.5))]
dataset = ImageDataset(opt.dataroot, transforms_=transforms_, data='CHAOS', get_name=False, unaligned=True)
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True,
                        pin_memory=True)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
# Make mask_H
mask_H, mask_L = make_mask([opt.batchSize, opt.input_nc, opt.size, int(opt.size / 2 + 1)], D=20)  # 10, 20, 30, 40, 50
freq_weight = 0
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
	sampler.set_epoch(epoch)
	for i, batch in enumerate(dataloader):
		# Set model input
		real_A = Variable(input_A.copy_(batch['A']))
		real_B = Variable(input_B.copy_(batch['B']))

		# Fourier Transform
		# real_A_freq = Fourier_trans(NonLinearLayer(real_A), mask_H, mask_L)
		# real_B_freq = Fourier_trans(NonLinearLayer(real_B), mask_H, mask_L)
		real_A_freq = Fourier_trans(real_A, mask_H, mask_L)
		real_B_freq = Fourier_trans(real_B, mask_H, mask_L)
		real_A_freq = init_transform(real_A_freq)
		real_B_freq = init_transform(real_B_freq)

		# Set Frequency loss weight
		if epoch > 50:
			freq_weight = 5.0
		else:
			freq_weight = 0.0

		###### Generators A2B and B2A ######
		optimizer_G.zero_grad()

		# GAN loss
		fake_B = netG_A2B(real_A)
		pred_fake = netD_B(fake_B)
		loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

		fake_A = netG_B2A(real_B)
		pred_fake = netD_A(fake_A)
		loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

		# FSeSim Loss
		loss_FSeSim = FSeSimLoss(real_A, fake_B) * 10.0 + FSeSimLoss(real_B, fake_A) * 10.0

		# Fourier Transform
		# fake_A_freq = Fourier_trans(NonLinearLayer(fake_A), mask_H, mask_L)
		# fake_B_freq = Fourier_trans(NonLinearLayer(fake_B), mask_H, mask_L)
		fake_A_freq = Fourier_trans(fake_A, mask_H, mask_L)
		fake_B_freq = Fourier_trans(fake_B, mask_H, mask_L)
		fake_A_freq = init_transform(fake_A_freq)
		fake_B_freq = init_transform(fake_B_freq)

		# Low frequency loss
		# freqL_A2B = netG_A2B(real_A_freq[1])
		# freqL_B2A = netG_B2A(real_B_freq[1])
		#
		# loss_freqL_B2A = criterion_identity(fake_A_freq[1], freqL_B2A)
		# loss_freqL_A2B = criterion_identity(fake_B_freq[1], freqL_A2B)
		# loss_freqL = loss_freqL_A2B + loss_freqL_B2A

		# High frequency loss
		freqH_A2B = real_A_freq[0]
		freqH_B2A = real_B_freq[0]
		# freqH_A2B = netG_A2B(real_A_freq[0])
		# freqH_B2A = netG_B2A(real_B_freq[0])

		loss_freqH_B2A = criterion_identity(fake_A_freq[0], freqH_B2A)
		loss_freqH_A2B = criterion_identity(fake_B_freq[0], freqH_A2B)
		loss_freqH = loss_freqH_A2B + loss_freqH_B2A

		loss_freq = (loss_freqH) * freq_weight

		# Total loss
		loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_FSeSim
		loss_G.backward()

		optimizer_G.step()
		###################################

		###### Discriminator A ######
		optimizer_D_A.zero_grad()

		# Real loss
		pred_real = netD_A(real_A)
		loss_D_real = criterion_GAN(pred_real, target_real)

		# Fake loss
		fake_A = fake_A_buffer.push_and_pop(fake_A)
		pred_fake = netD_A(fake_A.detach())
		loss_D_fake = criterion_GAN(pred_fake, target_fake)

		# Total loss
		loss_D_A = (loss_D_real + loss_D_fake) * 0.5
		loss_D_A.backward()

		optimizer_D_A.step()
		###################################

		###### Discriminator B ######
		optimizer_D_B.zero_grad()

		# Real loss
		pred_real = netD_B(real_B)
		loss_D_real = criterion_GAN(pred_real, target_real)

		# Fake loss
		fake_B = fake_B_buffer.push_and_pop(fake_B)
		pred_fake = netD_B(fake_B.detach())
		loss_D_fake = criterion_GAN(pred_fake, target_fake)

		# Total loss
		loss_D_B = (loss_D_real + loss_D_fake) * 0.5
		loss_D_B.backward()

		optimizer_D_B.step()
		###################################
		if local_rank == 0:
			# Progress report (http://localhost:8097)
			logger.log({'loss_G': loss_G,
			            'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
			            'loss_D': (loss_D_A + loss_D_B),
			            'loss_freq': loss_freq,
			            'loss_FSeSim': loss_FSeSim,
			            }
			           )

	# Update learning rates
	lr_scheduler_G.step()
	lr_scheduler_D_A.step()
	lr_scheduler_D_B.step()

	# Save models checkpoints
	if local_rank == 0:
		torch.save(netG_A2B.state_dict(), './output/' + str(epoch) + '_netG_A2B' + '.pth')
		torch.save(netG_B2A.state_dict(), './output/' + str(epoch) + '_netG_B2A' + '.pth')
		torch.save(netD_A.state_dict(), './output/' + str(epoch) + '_netD_A' + '.pth')
		torch.save(netD_B.state_dict(), './output/' + str(epoch) + '_netD_B' + '.pth')
###################################
