#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:54:11 2024

@author: xingw
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pathlib
import torch
import torch.nn as nn
# from utils import support
import math
import hdf5storage
#import os

def img_read(path, n_channels=3):
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img

def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
    
def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
def uint2single(img):

    return np.float32(img/255.)

def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())
    
# --------------------------------------------
# numpy(uint) (HxWxC or HxW) <--->  tensor
# --------------------------------------------


# convert uint to 4-dimensional torch tensor
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0) #


# convert uint to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)

def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)

def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    # img = np.expand_dims(img, axis=2)
    return np.uint8((img*255.0).round()) #

def tensor2uint_color(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
#    img = np.expand_dims(img, axis=2)
    return np.uint8((img*255.0).round()) 
    
# --------------------------------------------
# matlab's imwrite
# --------------------------------------------

def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)
    # plt.imsave(img_path, img)

###########Image Distoration##############
def kernel(mode, size):
    # global k
    if mode == 'ave':
        k = np.ones((size, size))
        k = k/np.sum(k)
    elif mode == 'gau':
        # --------------------------------
        # load kernel
        # --------------------------------
        kernels = hdf5storage.loadmat('kernels_12.mat')['kernels']
        k = kernels[0, size].astype(np.float64)
    elif mode == 'real':
        # --------------------------------
        # load kernel
        # --------------------------------
        kernels = hdf5storage.loadmat('Levin09.mat')['kernels']
        k = kernels[0, size].astype(np.float64)
    return k

def PSF(kernel, H, W):
    v = kernel
    ghy,ghx = v.shape
    psf = np.zeros((H, W, 1))
    psf[0:ghy, 0:ghx, 0] = v
    psf = np.roll(psf, -int(np.round((ghy-1)/2)), axis=0)
    psf = np.roll(psf, -int(np.round((ghx-1)/2)), axis=1)
    
    return psf

def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2,-1))
    #n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    #otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf

def pre_calculate(x, k):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    # b, c, w, h = x.shape#[-2:]
    w, h = x.shape[-2:]
    FB = p2o(k, (w, h))
    # FB = torch.ones(b, c, w*sf, h*sf).type_as(x)
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    FBFy = FBC*torch.fft.fftn(x, dim=(-2, -1))
    return FB, FBC, F2B, FBFy

def batch_PSF(K, H, W):
    Ks = K.detach().cpu().numpy()
    k1_size = Ks[0,0,0,0]
    k2_size = Ks[1,0,0,0]
    k1 = kernel('ave', int(k1_size))
    k2 = kernel('ave', int(k2_size))
    psf1 = PSF(k1, H, W)
    psf2 = PSF(k2, H, W)
    psf = np.zeros((2, H, W, 1))
    psf[0,:,:,:] = psf1
    psf[1,:,:,:] = psf2
    
    return psf

def batch_alpha(sigma):
    alpha = np.zeros((2,1,1,1))
    sigmas = sigma.detach().cpu().numpy()
    sigma1 = sigmas[0,0,0,0]
    sigma2 = sigmas[1,0,0,0]
    alpha1 = 0.8
    alpha2 = 0.8
    if sigma1 > 4.0:
        alpha1 = 0.7
    if sigma2 > 4.0:
        alpha2 = 0.7
        
    alpha[0,:,:,:] = alpha1
    alpha[1,:,:,:] = alpha2
    return alpha

def blur(img, k_test, n):
    img_L = ndimage.filters.convolve(img, np.expand_dims(k_test, axis=2), mode='wrap')
    np.random.seed(seed=0)  # for reproducibility
    img_L = uint2single(img_L)
    img_L = img_L+np.random.normal(0, n, img_L.shape) # add AWGN
    return np.float32(img_L)

def blur_train(img, k_train, n):
    img_L = ndimage.filters.convolve(img, np.expand_dims(k_train, axis=2), mode='wrap')
#    np.random.seed(seed=0)  # for reproducibility
    img_L = uint2single(img_L)
    img_L = img_L+np.random.normal(0, n, img_L.shape) # add AWGN
    return np.float32(img_L)

def noise_random(img, n):
#    img_L = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='wrap')
#    np.random.seed(seed=0)  # for reproducibility
    noise = np.random.normal(0, n, img.shape)
    img_N = img + noise# add AWGN
    return np.float32(img_N)

def loss_set(mode):
    if mode == 'L1':
        loss_fn = nn.L1Loss()
    # elif mode == 'HVS':
    #     loss_fn = support.HVS_gray_loss()
    return loss_fn

def csnr(A, B, row, col):
    n,m,ch = A.shape
    if ch == 1:
        e = A-B
        e = e[row:n-row-1, col:m-col-1]
        me=np.mean(np.mean(e**2))
        s=10*np.log10(255**2/me)
        return s
    else:
        e = A-B
        e = e[row:n-row-1, col:m-col-1, :]
        me1=np.mean(np.mean(e[:,:,0]**2))
        s1=10*np.log10(255**2/me1)
        me2=np.mean(np.mean(e[:,:,1]**2))
        s2=10*np.log10(255**2/me2)
        me3=np.mean(np.mean(e[:,:,2]**2))
        s3=10*np.log10(255**2/me3)
        return [s1, s2, s3]
    
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
    
def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img