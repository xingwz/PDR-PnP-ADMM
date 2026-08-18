#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 21:53:15 2025

@author: xingwz
"""

import torch.utils.data as data
import os
import glob
import numpy as np
import utils.util_image as util
import utils.utils_sisr as sr
import random
import torch
import cv2

class DataGenerator(data.Dataset):
    def __init__(self, data_path, sf, kernel, sigma, n_channels, mode='train'):
        super(DataGenerator, self).__init__()
        self.data_files = sorted(glob.glob(os.path.join(data_path,'*.*')))
        self.c = n_channels
        self.sf = sf
        self.k = kernel
        self.mode = mode
        self.map = sigma/255.
        self.patch_size = 256
        print('Number of acquired images = ',len(self.data_files))
    
    def __getitem__(self, index):
        img_path = self.data_files[index]
        img_H = util.img_read(img_path, self.c) # (0,255)
        H, W, _ = img_H.shape
        
        if self.mode == 'train':
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
        else:
            img_H = img_H[:self.patch_size, :self.patch_size, :]
        
        img_H = util.modcrop(img_H, np.lcm(self.sf,8))
        img_L = sr.classical_degradation(img_H, self.k, self.sf)
        img_L = util.uint2single(img_L)
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, self.map, img_L.shape) # add AWGN
        
        interp_L = cv2.resize(img_L, (img_L.shape[1]*self.sf, img_L.shape[0]*self.sf), interpolation=cv2.INTER_CUBIC)
        if np.ndim(interp_L)==2:
            interp_L = interp_L[..., None]

        interp_L = sr.shift_pixel2(interp_L, self.sf)
        
        img_H = util.uint2tensor3(img_H)
        img_L = util.single2tensor3(img_L)
        interp_L = util.single2tensor3(interp_L)
            
        noise_level = torch.FloatTensor([self.map]).view([1,1,1])
        return img_H, img_L, interp_L, noise_level
    
    def __len__(self):
        return len(self.data_files)