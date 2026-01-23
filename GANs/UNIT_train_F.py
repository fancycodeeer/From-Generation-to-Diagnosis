"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from UNIT_models import AdaINGen, MsImageDis, VAEGen, NonLinearLayer
from UNIT_utils import weights_init
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
from utils_b import ReplayBuffer
from utils_b import LambdaLR
from utils_b import Logger
from utils_b import Fourier_trans, init_transform, make_mask, AdaptiveFrequencyMask
from datasets_b import ImageDataset
import argparse
import itertools
import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import gc

# python -m torch.distributed.launch --nproc_per_node 8 train.py

def recon_criterion(input, target): # L1loss
    return torch.mean(torch.abs(input - target))

def __compute_kl(mu):
    # def _compute_kl(self, mu, sd):
    # mu_2 = torch.pow(mu, 2)
    # sd_2 = torch.pow(sd, 2)
    # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
    # return encoding_loss
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

def dis_calc_dis_loss(outs0, outs1, gan_type='lsgan'):
    # calculate the loss to train D
    loss = 0

    for it, (out0, out1) in enumerate(zip(outs0, outs1)):
        if gan_type == 'lsgan':
            loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
        elif gan_type == 'nsgan':
            all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                               F.binary_cross_entropy(F.sigmoid(out1), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(gan_type)
    return loss

def dis_calc_gen_loss(outs0, gan_type='lsgan'):
    # calculate the loss to train G
    loss = 0
    for it, (out0) in enumerate(outs0):
        if gan_type == 'lsgan':
            loss += torch.mean((out0 - 1)**2) # LSGAN
        elif gan_type == 'nsgan':
            all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
            loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(gan_type)
    return loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

################### MAIN #######################

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./datasets/CHAOS/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_false', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--multiGPU', type=bool, default=True, help='Use multiGPU train')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("--world_size", type=int, default=8)
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
if torch.cuda.is_available() and opt and opt.multiGPU:
    print("多GPU训练")
    set_seed(20230316)
    local_rank = opt.local_rank
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=local_rank)
    torch.cuda.set_device(local_rank)

# Initiate the networks
lr = opt.lr
gen_a = VAEGen(opt.input_nc, dim=64, n_downsample=2, n_res=4, activ='relu', pad_type='reflect')  # auto-encoder for domain a
gen_b = VAEGen(opt.output_nc, dim=64, n_downsample=2, n_res=4, activ='relu', pad_type='reflect')  # auto-encoder for domain b
dis_a = MsImageDis(opt.input_nc, n_layer=4, gan_type='lsgan', dim=64, norm='none', activ='lrelu', num_scales=3, pad_type='reflect')  # discriminator for domain a
dis_b = MsImageDis(opt.output_nc, n_layer=4, gan_type='lsgan', dim=64, norm='none', activ='lrelu', num_scales=3, pad_type='reflect') # discriminator for domain b
# NonLinearLayer = NonLinearLayer(opt.input_nc)
# instancenorm = nn.InstanceNorm2d(512, affine=False)
AdapFreMask = AdaptiveFrequencyMask(min_D=1, max_D=60)

# deploy models on GPUs
if opt.cuda:
    # opt.device = torch.device('cuda', local_rank)
    gen_a.cuda()
    gen_b.cuda()
    dis_a.cuda()
    dis_b.cuda()
    AdapFreMask.cuda()
    # NonLinearLayer.cuda()

if opt.multiGPU:
    gen_a = torch.nn.parallel.DistributedDataParallel(gen_a, device_ids=[local_rank], output_device=local_rank)
    gen_b = torch.nn.parallel.DistributedDataParallel(gen_b, device_ids=[local_rank], output_device=local_rank)
    dis_a = torch.nn.parallel.DistributedDataParallel(dis_a, device_ids=[local_rank], output_device=local_rank)
    dis_b = torch.nn.parallel.DistributedDataParallel(dis_b, device_ids=[local_rank], output_device=local_rank)
    AdapFreMask = torch.nn.parallel.DistributedDataParallel(AdapFreMask, device_ids=[local_rank], output_device=local_rank)
    # NonLinearLayer = torch.nn.parallel.DistributedDataParallel(NonLinearLayer, device_ids=[local_rank], output_device=local_rank)

# Setup the optimizers
beta1 = 0.5
beta2 = 0.999
dis_params = list(dis_a.parameters()) + list(dis_b.parameters())
gen_params = list(gen_a.parameters()) + list(gen_b.parameters()) # + list(NonLinearLayer.parameters())
AdapFreMask_params = list(AdapFreMask.parameters())

dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                lr=lr, betas=(beta1, beta2), weight_decay=0.0001)
gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                lr=lr, betas=(beta1, beta2), weight_decay=0.0001)
AdapFreMask_opt = torch.optim.Adam([p for p in AdapFreMask_params if p.requires_grad],
                                lr=lr, betas=(beta1, beta2), weight_decay=0.0001)

dis_scheduler = torch.optim.lr_scheduler.LambdaLR(dis_opt, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)# get_scheduler(dis_opt, hyperparameters)
gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_opt, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)# get_scheduler(gen_opt, hyperparameters)
AdapFreMask_scheduler = torch.optim.lr_scheduler.LambdaLR(AdapFreMask_opt, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Network weight initialization
gen_a.apply(weights_init('kaiming'))
gen_b.apply(weights_init('kaiming'))
# NonLinearLayer.apply(weights_init('kaiming'))
dis_a.apply(weights_init('gaussian'))
dis_b.apply(weights_init('gaussian'))
AdapFreMask.apply(weights_init('kaiming'))



# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_a = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_b = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
# target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
# target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
fake_a_buffer = ReplayBuffer()
fake_b_buffer = ReplayBuffer()
L1loss = torch.nn.L1Loss()

# Dataset loader
transforms_ = [ transforms.Grayscale(),
                transforms.Resize(int(opt.size*1.12), transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)) ]
dataset = ImageDataset(opt.dataroot, transforms_=transforms_, data='CHAOS', get_name=False)
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=(sampler is None), num_workers=opt.n_cpu, drop_last=True, sampler=sampler, pin_memory=True)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
# Make mask_H
# mask_H, mask_L = make_mask([opt.batchSize, opt.input_nc, opt.size, int(opt.size/2+1)], D=10)

############################# Train ###############################
for epoch in range(opt.epoch, opt.n_epochs):
    sampler.set_epoch(epoch)
    for i, batch in enumerate(dataloader):
        # Set model input
        x_a = Variable(input_a.copy_(batch['A']))
        x_b = Variable(input_b.copy_(batch['B']))

        # Make Mask
        mask_H_a, mask_L_a, D_a = AdapFreMask(x_a, [opt.batchSize, opt.input_nc, opt.size, int(opt.size/2+1)])
        mask_H_b, mask_L_b, D_b = AdapFreMask(x_b, [opt.batchSize, opt.input_nc, opt.size, int(opt.size/2+1)])

        # Fourier Transform
        # x_a_freq = Fourier_trans(NonLinearLayer(x_a), mask_H, mask_L)
        # x_b_freq = Fourier_trans(NonLinearLayer(x_b), mask_H, mask_L)
        x_a_freq = Fourier_trans(x_a, mask_H_a, mask_L_a)
        x_b_freq = Fourier_trans(x_b, mask_H_b, mask_L_b)
        x_a_freq = init_transform(x_a_freq)
        x_b_freq = init_transform(x_b_freq)

        ####### Generate ########
        gen_opt.zero_grad()
        # encode(within domain)
        h_a, n_a = gen_a(x_a, 'encode')
        h_b, n_b = gen_b(x_b, 'encode')
        # decode(within domain)
        x_a_recon = gen_a(h_a + n_a, 'decode')
        x_b_recon = gen_b(h_b + n_b, 'decode')
        # decode (cross domain)
        x_ba = gen_a(h_b + n_b, 'decode')
        x_ab = gen_b(h_a + n_a, 'decode')
        # encode again
        h_b_recon, n_b_recon = gen_a(x_ba, 'encode')
        h_a_recon, n_a_recon = gen_b(x_ab, 'encode')
        # decode again
        x_aba = gen_a(h_a_recon + n_a_recon, 'decode')
        x_bab = gen_b(h_b_recon + n_b_recon, 'decode')

        # Fourier Transform
        # x_ba_freq = Fourier_trans(NonLinearLayer(x_ba), mask_H, mask_L)
        # x_ab_freq = Fourier_trans(NonLinearLayer(x_ab), mask_H, mask_L)
        x_ba_freq = Fourier_trans(x_ba, mask_H_a, mask_L_a)
        x_ab_freq = Fourier_trans(x_ab, mask_H_b, mask_L_b)
        x_ba_freq = init_transform(x_ba_freq)
        x_ab_freq = init_transform(x_ab_freq)

        # encode(within domain)
        h_freq_a_H, n_freq_a_H = gen_a(x_a_freq[0], 'encode')
        h_freq_a_L, n_freq_a_L = gen_a(x_a_freq[1], 'encode')
        h_freq_b_H, n_freq_b_H = gen_b(x_b_freq[0], 'encode')
        h_freq_b_L, n_freq_b_L = gen_b(x_b_freq[1], 'encode')

        # decode(within domain)
        x_freq_a_recon_H = gen_a(h_freq_a_H + n_freq_a_H, 'decode')
        x_freq_b_recon_H = gen_b(h_freq_b_H + n_freq_b_H, 'decode')
        x_freq_a_recon_L = gen_a(h_freq_a_L + n_freq_a_L, 'decode')
        x_freq_b_recon_L = gen_b(h_freq_b_L + n_freq_b_L, 'decode')

        # decode (cross domain)
        x_freq_ba_H = gen_a(h_freq_b_H + n_freq_b_H, 'decode')
        x_freq_ab_H = gen_b(h_freq_a_H + n_freq_a_H, 'decode')
        x_freq_ba_L = gen_a(h_freq_b_L + n_freq_b_L, 'decode')
        x_freq_ab_L = gen_b(h_freq_a_L + n_freq_a_L, 'decode')

        # frequency loss
        # loss_gen_freq_x_a_recon = L1loss(x_freq_a_recon_L, x_a_freq[1]) + L1loss(x_freq_a_recon_H, x_a_freq[0])
        # loss_gen_freq_x_b_recon = L1loss(x_freq_b_recon_L, x_b_freq[1]) + L1loss(x_freq_b_recon_H, x_b_freq[0])

        loss_gen_freq_x_ba = L1loss(x_freq_ba_L, x_ba_freq[1]) + L1loss(x_freq_ba_H, x_ba_freq[0]) # + L1loss(x_freq_ba_H, x_b_freq[0])
        loss_gen_freq_x_ab = L1loss(x_freq_ab_L, x_ab_freq[1]) + L1loss(x_freq_ab_H, x_ab_freq[0]) # + L1loss(x_freq_ab_H, x_a_freq[0])
        loss_gen_freq_kl = __compute_kl(h_freq_a_H) + __compute_kl(h_freq_a_L) + __compute_kl(h_freq_b_H) + __compute_kl(h_freq_b_L)

        # FDIT
        # loss_gen_freq_x_ba = L1loss(x_ba_freq[0], x_b_freq[0])
        # loss_gen_freq_x_ab = L1loss(x_ab_freq[0], x_a_freq[0])
        # loss_gen_freq_kl = __compute_kl(h_freq_a_H) + __compute_kl(h_freq_a_L) + __compute_kl(h_freq_b_H) + __compute_kl(h_freq_b_L)

        # reconstruction loss
        loss_gen_recon_x_a = recon_criterion(x_a_recon, x_a)
        loss_gen_recon_x_b = recon_criterion(x_b_recon, x_b)
        loss_gen_recon_kl_a = __compute_kl(h_a)
        loss_gen_recon_kl_b = __compute_kl(h_b)
        loss_gen_cyc_x_a = recon_criterion(x_aba, x_a)
        loss_gen_cyc_x_b = recon_criterion(x_bab, x_b)
        loss_gen_recon_kl_cyc_aba = __compute_kl(h_a_recon)
        loss_gen_recon_kl_cyc_bab = __compute_kl(h_b_recon)

        # GAN loss
        loss_gen_adv_a = dis_calc_gen_loss(dis_a(x_ba))
        loss_gen_adv_b = dis_calc_gen_loss(dis_b(x_ab))

        # total loss
        loss_gen_total = 1.0 * loss_gen_adv_a + \
                         1.0 * loss_gen_adv_b + \
                         10.0 * loss_gen_recon_x_a + \
                         0.01 * loss_gen_recon_kl_a + \
                         10.0 * loss_gen_recon_x_b + \
                         0.01 * loss_gen_recon_kl_b + \
                         10.0 * loss_gen_cyc_x_a + \
                         0.01 * loss_gen_recon_kl_cyc_aba + \
                         10.0 * loss_gen_cyc_x_b + \
                         0.01 * loss_gen_recon_kl_cyc_bab + \
                         10.0 * loss_gen_freq_x_ba + \
                         10.0 * loss_gen_freq_x_ab + \
                         0.01 * loss_gen_freq_kl # + \
                         # 10.0 * loss_gen_freq_x_a_recon + \
                         # 10.0 * loss_gen_freq_x_b_recon

        loss_gen_total.backward()
        gen_opt.step()


        ####### Discriminator ########
        dis_opt.zero_grad()
        x_ba = fake_a_buffer.push_and_pop(x_ba)
        x_ab = fake_b_buffer.push_and_pop(x_ab)
        # D loss
        loss_dis_a = dis_calc_dis_loss(dis_a(x_ba.detach()), dis_a(x_a))
        loss_dis_b = dis_calc_dis_loss(dis_b(x_ab.detach()), dis_b(x_b))
        loss_dis_total = 1.0 * loss_dis_a + 1.0 * loss_dis_b

        loss_dis_total.backward()
        dis_opt.step()

        ######### Logger ###########
        if local_rank == 0:
            # Progress report (http://localhost:8097)
            logger.log({'loss_G': loss_gen_total,
                        'loss_G_GAN': loss_gen_adv_a + loss_gen_adv_b,
                        'loss_D': loss_dis_a + loss_dis_b,
                        'loss_cyc': loss_gen_cyc_x_a + loss_gen_cyc_x_b,
                        'loss_recon': loss_gen_recon_x_a + loss_gen_recon_x_b,
                        'loss_freq': # loss_gen_freq_x_a_recon + loss_gen_freq_x_b_recon + \
                                     loss_gen_freq_x_ba + loss_gen_freq_x_ab,
                        })

        # release GPU memory
        # gc.collect()
        # torch.cuda.empty_cache()

    # Update learning rates
    dis_scheduler.step()
    gen_scheduler.step()

    # Save models checkpoints
    if local_rank == 0:
        torch.save(gen_a.module.state_dict(), './output/' + str(epoch) + '_netG_A' + '.pth')
        torch.save(gen_b.module.state_dict(), './output/' + str(epoch) + '_netG_B' + '.pth')
        torch.save(dis_a.module.state_dict(), './output/' + str(epoch) + '_netD_A' + '.pth')
        torch.save(dis_b.module.state_dict(), './output/' + str(epoch) + '_netD_B' + '.pth')
        # torch.save(NonLinearLayer.module.state_dict(), './output/' + str(epoch) + '_UNIT_NL' + '.pth')

    # gc.collect()
    # torch.cuda.empty_cache()
#######################################################################
