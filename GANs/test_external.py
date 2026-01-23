#!/usr/bin/python3
import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import time
import numpy as np
import random
from models import Generator
from datasets_b import ImageDataset
from SSIM_PSNR import Evaluation_function, FID_score


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='E:\BraTS2024\外部性检验/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--size', type=int, default=240, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_false', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='D:\CT2MR\T2_output\Pix2pix+FDIT/prograssive/299_netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='D:\CT2MR\T2_output\Pix2pix+FDIT/prograssive/299_netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--output_dir', type=str, default='E:\BraTS2024\外部性检验/T22Flair/p2p+fdit/299')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
# netG_A2B = torch.nn.DataParallel(Generator(opt.input_nc, opt.output_nc))
# netG_B2A = torch.nn.DataParallel(Generator(opt.output_nc, opt.input_nc))
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
# NonLinearLayer = NonLinearLayer(opt.input_nc)


if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    # NonLinearLayer.cuda()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.Grayscale(),
                transforms.Resize((opt.size, opt.size)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                transforms.Normalize((0.5), (0.5))
              ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_,
                                     mode='test',
                                     data=None,
                                     get_name=True,
                                     unaligned=False),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
print(len((dataloader)))

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
# NonLinearLayer.load_state_dict(torch.load(opt.Nonlinear))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()
# NonLinearLayer.eval()
###################################

###### Testing######
if not os.path.exists(os.path.join(opt.output_dir, 'A')):
    os.makedirs(os.path.join(opt.output_dir, 'A'))
if not os.path.exists(os.path.join(opt.output_dir, 'B')):
    os.makedirs(os.path.join(opt.output_dir, 'B'))

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))
    name_A = batch['name_A'][0]
    name_B = batch['name_B'][0]


    # Generate output
    start_time = time.time()
    fake_B = 0.5*(netG_A2B((real_A)).data + 1.0)
    fake_B = nn.functional.interpolate(fake_B, (256, 256))
    fake_A = 0.5*(netG_B2A((real_B)).data + 1.0)
    fake_A = nn.functional.interpolate(fake_A, (256, 256))
    end_time = time.time()
    print(end_time - start_time)

    # Save image files
    save_image(fake_A, os.path.join(opt.output_dir, 'A') + '/' + name_A)
    save_image(fake_B, os.path.join(opt.output_dir, 'B') + '/' + name_B)


    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))


sys.stdout.write('\n')
###################################
