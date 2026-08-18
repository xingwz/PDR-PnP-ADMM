#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 22:21:47 2025

@author: xingwz
"""

import argparse
import torch
import utils.util_image as util
from model.network_unet import UNetRes as net
from model.network_H import Allin1_HyPaNet
import numpy as np
import torch.optim as optim
from data_sisr import DataGenerator
from torch.utils.data import DataLoader
import random
from utils.training_sisr import train_sisr
#try
parser = argparse.ArgumentParser(description='HyperNet *^_^*')
parser.add_argument('--img_path', type=str, required=False, dest='img_path',
                    default='data/kodak/')
parser.add_argument('--valid_path', type=str, required=False, dest='valid_path',
                    default='data/set5/')
parser.add_argument('--log_path', type=str, required=False, dest='log_path',
                    default='log/SISR_PnP/R_ADMM_Image/')
parser.add_argument('--alpha', required=False, action='store_true', default=False, dest='alpha')
parser.add_argument('--mode', type=str, required=False, dest='mode', default='perImage')
parser.add_argument('--n_channels', type=int, required=False, default=1, dest='n_channels')
parser.add_argument('--batch_size', type=int, required=False, default=2, dest='batch_size')
parser.add_argument('--epochs', type=int, required=False, default=500, dest='num_epochs')
parser.add_argument('--logloss', type=int, required=False, default=1, dest='log_loss_every')
parser.add_argument('--logdisp', type=int, required=False, default=100, dest='log_display_every')
parser.add_argument('--iter_num', type=int, required=False, dest='iter_num', default=8)
parser.add_argument('--scale_factor', type=int, required=False, dest='scale_factor', default=2)
parser.add_argument('--kernel_size', type=int, required=False, dest='kernel_size', default=0)
parser.add_argument('--kernel_mode', type=str, required=False, dest='kernel_mode', default='real')
parser.add_argument('--noise_level', type=float, required=False, dest='noise_level', default=0)
parser.add_argument('--loss', type=str, required=False, dest='loss', default='L1')
parser.add_argument('--show_img', required=False, action='store_true', default=False, dest='show_img')
parser.add_argument('--gpu', required=False, action='store_true', default=True, dest='use_gpu')
parser.add_argument('--device', type=str, required=False, default='cuda:0')
config = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

seed = random.randint(1, 10000)
print('Random seed: {}'.format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#######DRUNet load########
n_channels = config.n_channels
if n_channels == 1:
    mode = 'gray_'
    checkpoint = torch.load('model_zoo/drunet_gray.pth', map_location="cuda:0")
else:
    mode = 'color_'
    checkpoint = torch.load('model_zoo/drunet_color.pth', map_location="cuda:0")
model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
model.load_state_dict(checkpoint, strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
#######HyperNet load########
if config.mode == 'perIter':
    HyperNet = Allin1_HyPaNet(in_nc=2, out_nc=config.iter_num, alpha=config.alpha)
else:
    HyperNet = Allin1_HyPaNet(in_nc=2, out_nc=1, alpha=config.alpha)
HyperNet = HyperNet.to(device)

HyperNet.train()
optimizer_H = optim.Adam(HyperNet.parameters(),lr=1e-4)

#### Data Generation ####
sf = config.scale_factor
## kernel ##
k = util.kernel(config.kernel_mode, config.kernel_size)
util.imshow(k) if config.show_img else None
print('Creating database...')
# data_path, kernel, sigma, n_channels
dataset = DataGenerator(data_path=config.img_path, sf=sf, kernel=k, sigma=config.noise_level, n_channels=n_channels, mode ='train')
dataloader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True, num_workers=10)
print('done...')
print('Creating valid data...')
validset = DataGenerator(data_path=config.valid_path, sf=sf, kernel=k, sigma=config.noise_level, n_channels=n_channels, mode = 'valid')
validloader = DataLoader(validset,batch_size=1,shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
print('done...')
print('Creating variables...')
logs_root_path = config.log_path+ mode+'sf'+str(config.scale_factor)+'_'+config.kernel_mode + str(config.kernel_size)+'_n'+str(config.noise_level)
print('done...')

loss_fn = util.loss_set(config.loss)

train_sisr(h=k, sf=sf, Opts=config, model=model, device=device, 
           logs_root_path=logs_root_path, dataloader=dataloader, 
           validloader=validloader, HyperNet=HyperNet, #netE=netE, 
           optimizer_H=optimizer_H, loss_fn=loss_fn)

