#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 14:05:43 2025

@author: xingwz
"""

import argparse
import torch
import glob
import os
import utils.util_image as util
import utils.utils_sisr as sr
from model.network_unet import UNetRes as net
import numpy as np
from ttictoc import tic,toc
import cv2
import logging
from utils import utils_logger, utils_hyps

import matplotlib.pyplot as plt
import scipy.io

parser = argparse.ArgumentParser(description='SISR PDR_PnP_ADMM_Relaxation perIter*^_^*')
parser.add_argument('--img_path', type=str, required=False, dest='img_path',
                    default='data/CBSD68/')
parser.add_argument('--reuslt_path', type=str, required=False, dest='reuslt_path',
                    default='results/SISR_PnP/R_ADMM_Iter/')
parser.add_argument('--mode', type=str, required=False, dest='mode', default='NoR')
parser.add_argument('--n_channels', type=int, required=False, default=1, dest='n_channels')
parser.add_argument('--iter_num', type=int, required=False, dest='iter_num', default=8) # ADMM_Relax_H3
parser.add_argument('--scale_factor', type=int, required=False, dest='sf', default=2)
parser.add_argument('--kernel_size', type=int, required=False, dest='kernel_size', default=2)
parser.add_argument('--kernel_mode', type=str, required=False, dest='kernel_mode', default='gau')
parser.add_argument('--noise_level', type=float, required=False, dest='noise_level', default=0.0)
parser.add_argument('--show_img', required=False, action='store_true', default=False, dest='show_img')
parser.add_argument('--gpu', required=False, action='store_true', default=False, dest='use_gpu')
parser.add_argument('--device', type=str, required=False, default='cuda:0')
config = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

#######DRUNet load########
n_channels = config.n_channels
if n_channels == 1:
    mode = 'gray_'
    checkpoint = torch.load('model_zoo/drunet_gray.pth', map_location=device)
else:
    mode = 'color_'
    checkpoint = torch.load('model_zoo/drunet_color.pth', map_location=device)
    
reuslt_path = config.reuslt_path +mode+'sf' +str(config.sf)+'_k_' +config.kernel_mode+str(config.kernel_size) +'_n_'+str(config.noise_level)+'/'
util.mkdir(reuslt_path)
test_sf = [1,2,3,4]

logger_name = 'test'
utils_logger.logger_info(logger_name, log_path=os.path.join(reuslt_path, logger_name+'.log'))
logger = logging.getLogger(logger_name)
logger.info(utils_logger.dict2str(vars(config)))

model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
model.load_state_dict(checkpoint, strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
   
#### Step 1. Image preprocessing ####
data_files = sorted(glob.glob(os.path.join(config.img_path,'*.*')))
psnr_list = []
# tic()

for i in range(len(data_files)):#
    ### Step 1.1 load image, set kernel ###
    m = data_files[i].split('/')
#    print(i+1, ',process image:',m[-1])
    ## high quality image ##
    img_H = util.img_read(data_files[i], n_channels) # (0,255)
    ## kernel ##
    k = util.kernel(config.kernel_mode, config.kernel_size)
    util.imshow(k) if config.show_img else None
    ### Step 1.2 image distoration: y = k*x + n ###
    ## low quality image ##
    img_H = util.modcrop(img_H, np.lcm(config.sf,8))
    img_L = sr.classical_degradation(img_H, k, config.sf)
    img_L = util.uint2single(img_L)
    np.random.seed(seed=0)  # for reproducibility
    image_noise_level = config.noise_level/255.
    img_L += np.random.normal(0, image_noise_level, img_L.shape)
    util.imshow(img_L) if config.show_img else None
    util.imwrite(util.single2uint(img_L), reuslt_path+'LR_'+m[-1][:-4]+'.png')
    if n_channels == 1:
        img_H = img_H.squeeze()
    
    interp_L = cv2.resize(img_L, (img_L.shape[1]*config.sf, img_L.shape[0]*config.sf), interpolation=cv2.INTER_CUBIC)
    if np.ndim(interp_L)==2:
        interp_L = interp_L[..., None]
    interp_L = sr.shift_pixel2(interp_L, config.sf)
    
    interp_L = util.single2tensor4(interp_L).to(device)
    img_L_tensor, k_tensor = util.single2tensor4(img_L).to(device), util.single2tensor4(np.expand_dims(k, 2)).to(device)
    
    #### Step 2. Image SISR ####
    FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, config.sf)
    # Hyper-parameters
    mu, thr1, alpha = utils_hyps.hyper_params_color(config.kernel_size, config.sf, config.noise_level, config.mode)
    mu = torch.FloatTensor([mu]).view([1,8,1,1]) #
    thr1 = torch.FloatTensor([thr1]).view([1,8,1,1])
    mu_tensor, thr1_tensor = mu.to(device), thr1.to(device)
    # initialize
    z = interp_L
    x = z
    u = z*0
    psnr_iter = []
    for j in range(config.iter_num):

        # --------------------------------
        # step 1, X update
        # --------------------------------

        x = sr.data_solution((z-u).float(), FB, FBC, F2B, FBFy, mu_tensor[:,j,:,:], config.sf)
        x = (1-alpha[j])*(z-u) + alpha[j]*x    #(z-u)
    
        # --------------------------------
        # step 2, W (Z) update
        # --------------------------------
        
        x_u_sigma = torch.cat(((x+u).float(), thr1_tensor[:,j,:,:].repeat(1, 1, interp_L.shape[2], interp_L.shape[3])), dim=1)
        x_u_sigma = x_u_sigma.to(device)
        z = model(x_u_sigma)
        
        # --------------------------------
        # step 3, C update
        # --------------------------------
        u = u + (x - z)
        cur_psnr = util.calculate_psnr(util.tensor2uint(x), img_H, border=config.sf)
        print(f"[{j}]"," PSNR= ", cur_psnr)
        
    img_E = util.tensor2uint(x)
    psnr = util.calculate_psnr(img_E, img_H, border=config.sf)
    logger.info('{:->4d}--> {:>10s} -- x{:>2d} PSNR: {:.2f}dB'.format(i, m[-1], config.sf, psnr))
    psnr_list.append(psnr)
    util.imshow(img_E) if config.show_img else None
    util.imwrite(img_E, reuslt_path+'sisr_'+m[-1][:-4]+'_'+str(psnr)[:7]+'.png')
    print('***************************************')
    
# print(toc())
print('***************************************')
print("Average PSNR of dataset = ", sum(psnr_list)/len(data_files))
testset_name=config.img_path
testset_name=testset_name.split('/')[-2]
logger.info('------> Average PSNR(RGB) of ({}) scale factor: ({}), kernel: ({}) sigma: ({}): {:.2f} dB'.format(testset_name, config.sf, config.kernel_mode+str(config.kernel_size), config.noise_level, sum(psnr_list)/len(data_files)))


