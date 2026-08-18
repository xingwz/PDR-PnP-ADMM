#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 11:32:46 2025

@author: xingwz
"""

import argparse
import torch
import utils.util_image as util
from model.init_network import init_weights
import numpy as np
import torch.optim as optim
from dataset_usrnet import DatasetUSRNet
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import random
from utils.training_unfold import train_unfold
from torch.optim import lr_scheduler
from model.model_base import load_optimizer

#try
parser = argparse.ArgumentParser(description='All-in-one Stage 1 *^_^*')
parser.add_argument('--img_path', type=str, required=False, dest='img_path',
                    default='data/DIV2K_train_HR/')
parser.add_argument('--valid_path', type=str, required=False, dest='valid_path',
                    default='data/set5/')
parser.add_argument('--log_path', type=str, required=False, dest='log_path',
                    default='log/URADMMNet_SISR_')
parser.add_argument('--n_channels', type=int, required=False, default=1, dest='n_channels')
parser.add_argument('--batch_size', type=int, required=False, default=48, dest='batch_size')
parser.add_argument('--epochs', type=int, required=False, default=500, dest='num_epochs')
parser.add_argument('--logloss', type=int, required=False, default=1, dest='log_loss_every')
parser.add_argument('--logdisp', type=int, required=False, default=100, dest='log_display_every')
parser.add_argument('--iter_num', type=int, required=False, dest='iter_num', default=6)
parser.add_argument('--scale_factor', type=int, required=False, dest='scale_factor', default=2)
parser.add_argument('--noise_level', type=float, required=False, dest='noise_level', default=25.0)
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
    mode = 'gray'
else:
    mode = 'color'
if 'SRNet' in config.log_path:
    from model.network_usrnet_v1 import USRNet as net
elif 'ADMM' in config.log_path:
    from model.network_usrnet_v1 import URADMMNet as net
model = net(n_iter=config.iter_num, h_nc=32, in_nc=n_channels+1, out_nc=n_channels, nc=[16, 32, 64, 64],
            nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
model = model.to(device)
init_weights(model, init_type="orthogonal", init_bn_type="uniform", gain=0.2)
model = DataParallel(model)
model.train()
optimizer = optim.Adam(model.parameters(),lr=1e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, [100000, 200000, 300000, 400000], 0.5)

#### Data Generation ####
sf = config.scale_factor
print('Creating database...')
# data_path, kernel, sigma, n_channels
dataset = DatasetUSRNet(data_path=config.img_path, sf=sf, sigma=config.noise_level, n_channels=n_channels, batch_size=config.batch_size, mode='train')
dataloader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True, num_workers=10)
print('done...')
print('Creating valid data...')
validset = DatasetUSRNet(data_path=config.valid_path, sf=sf, sigma=config.noise_level, n_channels=n_channels, batch_size=1, mode='valid')
validloader = DataLoader(validset,batch_size=1,shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
print('done...')
print('Creating variables...')
logs_root_path = config.log_path+ mode
print('done...')

loss_fn = util.loss_set(config.loss)

train_unfold(Opts=config, model=model, device=device, 
             logs_root_path=logs_root_path, dataloader=dataloader, 
             validloader=validloader, optimizer=optimizer,
             scheduler=scheduler, loss_fn=loss_fn.to(device))

