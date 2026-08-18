#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 12:19:52 2025

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
from model.model_base import save_optimizer

def train_unfold(Opts, model, device, logs_root_path, dataloader, validloader, optimizer, scheduler, loss_fn):
    events_dir = logs_root_path+'/events/'
    image_dir = logs_root_path+'/images/'
    checkpoints_dir = logs_root_path+'/checkpoints/'
    valid_dir = logs_root_path+'/valid/'
    logloss = Opts.log_loss_every
    logdisp = Opts.log_display_every
    pathlib.Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(valid_dir).mkdir(parents=True, exist_ok=True)
    log_path = Opts.log_path
    module_name = log_path.split('/')[1]
    module_name = 'model_'+module_name.split('_')[0]
    
    print("Create summaries")
    writer = SummaryWriter(events_dir)
    ####Add by wenzhu####
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(events_dir, logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(utils_logger.dict2str(vars(Opts)))
    
    print("Start training")
    loss_avg = 0.0
    total_steps = 0
    current_step = 0
    
    with tqdm(total=len(dataloader) * Opts.num_epochs) as pbar:
        for epoch in range(Opts.num_epochs):
           
            for n, train_data in enumerate(dataloader):
                current_step += 1
                scheduler.step(current_step)
                
                L = train_data['L'].to(device)  # low-quality image
                k = train_data['k'].to(device)  # blur kernel
                sf = int(train_data['sf'][0,...].squeeze().cpu().numpy()) # scale factor
                sigma = train_data['sigma'].to(device)  # noise level
                H = train_data['H'].to(device)
                
                optimizer.zero_grad()
                E = model(L, k, sf, sigma)
                
                if n%10 == 0:
                    util.imwrite(util.tensor2uint(E[0,:,:,:]), image_dir+str(int(n)+1)+'_xpred.png')
                    
                
                loss = loss_fn(E, H)
                loss.backward()
                optimizer.step()
                
                loss_avg += support.psnr(E, H).cpu().detach()
                total_steps += 1
                
                if current_step%(len(dataloader)*logdisp)==0:
                    save_filename = '{}_{}.pth'.format(module_name, str(epoch+1))
                    save_path = os.path.join(checkpoints_dir, save_filename)
                    torch.save(model.state_dict(), save_path)
                    save_optimizer(checkpoints_dir, optimizer, 'optimizer', current_step)
                    ####Add by wenzhu####
                    logger.info('Saving the model.')
                if current_step%(len(dataloader)*logdisp)==0:
                    valid_psnr = 0.0
                    valid_num = 0
                    for n, valid_data in enumerate(validloader):
                        image_name_ext = os.path.basename(valid_data['L_path'][0])
                        img_name, ext = os.path.splitext(image_name_ext)
                        
                        L = valid_data['L'].to(device)  # low-quality image
                        k = valid_data['k'].to(device)  # blur kernel
                        sf = int(valid_data['sf'][0,...].squeeze().cpu().numpy()) # scale factor
                        sigma = valid_data['sigma'].to(device)  # noise level
                        H = valid_data['H'].to(device)
                        
                        model.eval()
                        with torch.no_grad():
                            E = model(L, k, sf, sigma)
                        model.train()
                        visual_E = E.detach()[0].float().cpu()
                        visual_H = H.detach()[0].float().cpu()
                        E_img = util.tensor2uint(visual_E)
                        H_img = util.tensor2uint(visual_H)
                        
                        save_img_path = os.path.join(valid_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                        util.imwrite(E_img, save_img_path)
                        psnr = util.calculate_psnr(E_img, H_img, border=Opts.scale_factor)
                        valid_psnr += psnr
                        valid_num += 1
                        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(valid_num, image_name_ext, psnr))

                    logger.info('<Validation: epoch:{:3d}, Average PSNR : {:<.2f}dB\n'.format((epoch+1), valid_psnr/valid_num))
                if current_step%(len(dataloader)*logloss)==0:
                    logger.info('<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, Average PSNR : {:<.2f}dB\n'.format((epoch+1), current_step, scheduler.get_lr()[0], loss_avg/total_steps))
            pbar.update(len(dataloader))
            print(f"[{epoch}]"," PSNR=", loss_avg/total_steps)
            loss_avg = 0.0
            total_steps = 0
    logger.info('End of training.')
                        
                        
                
                

                