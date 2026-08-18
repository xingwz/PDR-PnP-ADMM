#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:35:01 2025

@author: xingwz
"""

import pathlib
import os
import logging
from utils import utils_logger
from utils import support
from tqdm.autonotebook import tqdm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils.util_image as util
from utils import utils_model
from utils import utils_sisr as sr
from model.model_base import save_network
# from utils.support import stability_loss_monotone

def train_sisr(h, sf, Opts, model, device, logs_root_path, dataloader, validloader, HyperNet, optimizer_H, loss_fn):
    events_dir = logs_root_path+'/events/'
    image_dir = logs_root_path+'/images/'
    checkpoints_dir = logs_root_path+'/checkpoints/'
    valid_dir = logs_root_path+'/valid/'
    logloss = Opts.log_loss_every
    logdisp = Opts.log_display_every
    pathlib.Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(valid_dir).mkdir(parents=True, exist_ok=True)
    
    print("Create summaries")
    writer = SummaryWriter(events_dir)
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(events_dir, logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(utils_logger.dict2str(vars(Opts)))
    
    print("Start training")
    loss_avg = 0.0
    total_steps = 0
    current_step = 0
    
    IterNums = Opts.iter_num
    mode = Opts.mode
    
    with tqdm(total=len(dataloader) * Opts.num_epochs) as pbar:
        for epoch in range(Opts.num_epochs):
           
            for n, (data, blur, initial, sigma) in enumerate(dataloader,0):

                if Opts.use_gpu:
                    data = data.to(device) # BxCxHxsfxWxsf
                    blur = blur.to(device) # BxCxHxW
                    initial = initial.to(device) # BxCxHxsfxWxsf
                    sigma = sigma.to(device)              
                
                k_tensor = util.single2tensor4(np.expand_dims(h, 2))
                FB, FBC, F2B, FBFy = sr.pre_calculate(blur, k_tensor.to(device), sf)
                
                z = initial
                x = z
                u = z*0
                
                sf_tensor = torch.FloatTensor([sf]).view([1,1,1,1]).repeat(2,1,1,1)
                
                for param in HyperNet.parameters():
                    param.requires_grad = True
                optimizer_H.zero_grad()
                HyperNet.train()
                
                params = HyperNet(sf_tensor.to(device), sigma)
                for param in model.parameters():
                    param.requires_grad = False
                
                if mode == 'perImage':
                    mu = params[:, 0:1, ...].float()
                    thr = params[:, 1:2, ...].float()
                    if Opts.alpha:
                        alpha = params[:, 2:3, ...].float()
                for i in range(IterNums):
                    if mode == 'perIter':
                        mu = params[:, i:i+1, ...].float()
                        thr = params[:, i+IterNums:i+IterNums+1, ...].float()
                        if Opts.alpha:
                            alpha = params[:, i+IterNums*2:i+IterNums*2+1, ...].float()
                    x = sr.data_solution((z-u).float(), FB, FBC, F2B, FBFy, mu, sf)
                    if Opts.alpha:
                        x = (1-alpha)*(z-u) + alpha*x 
                    x_u_sigma = torch.cat(((x+u).float(), thr.repeat(1, 1, initial.shape[2], initial.shape[3])), dim=1)
                    if Opts.use_gpu:
                        x_u_sigma = x_u_sigma.to(device)
                    z = model(x_u_sigma)
                    u = u + (x - z)
                x_pred = x
                
                if n%10 == 0:
                    util.imwrite(util.tensor2uint(x_pred[0,:,:,:]), image_dir+str(int(n)+1)+'_xpred.png')
                
                loss = loss_fn(x_pred, data)# + 0.1*ls
                loss.backward()
                ''' updates of optimizer '''
                optimizer_H.step()
                
                ''' EMA '''
                # update_E(HyperNet, netE)
                
                loss_avg = loss_avg+support.psnr(x_pred, data).cpu().detach()
                total_steps += 1
                
                current_step += 1
                if current_step%(len(dataloader)*Opts.num_epochs)==0:
                    save_network(checkpoints_dir, HyperNet, 'G', str(epoch+1))
                    # save_network(checkpoints_dir, netE, 'E', str(epoch+1))
                    ####Add by wenzhu####
                    logger.info('Saving the model.')
                if current_step%(len(dataloader)*logdisp)==0:
                    valid_psnr = 0.0
                    valid_num = 0
                    for n, (data, blur, initial, sigma) in enumerate(validloader,0):

                        if Opts.use_gpu:
                            data = data.to(device) # BxCxHxsfxWxsf
                            blur = blur.to(device) # BxCxHxW
                            initial = initial.to(device) # BxCxHxsfxWxsf
                            sigma = sigma.to(device)              
                        
                        k_tensor = util.single2tensor4(np.expand_dims(h, 2))
                        FB, FBC, F2B, FBFy = sr.pre_calculate(blur, k_tensor.to(device), sf)
                        
                        z = initial
                        x = z
                        u = z*0
                        
                        sf_tensor = util.single2tensor4(torch.FloatTensor([sf]).view([1,1,1]))
                        
                        params = HyperNet(sf_tensor.to(device), sigma)
                        if epoch == Opts.num_epochs-1 and n==0:
                            # paramsE = netE(sf_tensor.to(device), sigma)
                            print(params)
                            # print(paramsE)
                        
                        if mode == 'perImage':
                            mu = params[:, 0:1, ...].float()
                            thr = params[:, 1:2, ...].float()
                            if Opts.alpha:
                                alpha = params[:, 2:3, ...].float()
                        for i in range(IterNums):
                            if mode == 'perIter':
                                mu = params[:, i:i+1, ...].float()
                                thr = params[:, i+IterNums:i+IterNums+1, ...].float()
                                if Opts.alpha:
                                    alpha = params[:, i+IterNums*2:i+IterNums*2+1, ...].float()
                            x = sr.data_solution((z-u).float(), FB, FBC, F2B, FBFy, mu, sf)
                            if Opts.alpha:
                                x = (1-alpha)*(z-u) + alpha*x
                            
                            x_u_sigma = torch.cat(((x+u).float(), thr.repeat(1, 1, initial.shape[2], initial.shape[3])), dim=1)
                            if Opts.use_gpu:
                                x_u_sigma = x_u_sigma.to(device)
                            z = model(x_u_sigma)
                    
                            if n%50 == 0:
                                logger.info('<Validation: epoch:{:3d}, image:{:2d}, iter:{:2d}, PSNR : {:<.2f}dB\n'.format((epoch+1), n+1, i+1, support.psnr(x, data).cpu().detach()))
                                if epoch == Opts.num_epochs-1:
                                    util.imwrite(util.tensor2uint(x[0,:,:,:]), valid_dir+'x_'+str(i+1)+'.png')
                                    util.imwrite(util.tensor2uint(z[0,:,:,:]), valid_dir+'z_'+str(i+1)+'.png')
#                                
                            u = u + (x - z)
                        x_pred = x
                        if epoch == Opts.num_epochs-1:
                            util.imwrite(util.tensor2uint(x_pred[0,:,:,:]), valid_dir+'epoch'+str(epoch+1)+'_'+str(int(n)+1)+'.png')
                        valid_psnr = valid_psnr+ support.psnr(x_pred, data).cpu().detach()
                        valid_num += 1
                    logger.info('<Validation: epoch:{:3d}, Average PSNR : {:<.2f}dB\n'.format((epoch+1), valid_psnr/valid_num))
                if current_step%(len(dataloader)//logloss)==0:
                    logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB\n'.format((epoch+1), current_step, loss_avg/total_steps))
            pbar.update(len(dataloader))
            print(f"[{epoch}]"," PSNR=", loss_avg/total_steps)
            loss_avg = 0.0
            total_steps = 0
    logger.info('End of training.')