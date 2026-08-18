import os.path
import cv2
import logging
import time
import os

import numpy as np
from datetime import datetime
from collections import OrderedDict
from scipy.io import loadmat
import hdf5storage
from scipy import ndimage
from scipy.signal import convolve2d

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import util_sisr as sr
from utils import utils_image as util
#from models.network_usrnet import USRNet as net   # for pytorch version <= 1.7.1
  # for pytorch version >= 1.8.1
from torch.nn.parallel import DataParallel
import argparse


'''
Spyder (Python 3.6)
PyTorch 1.4.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/USRNet
        https://github.com/cszn/KAIR

If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)

by Kai Zhang (12/March/2020)
'''

"""
# --------------------------------------------
testing code of USRNet for the Table 1 in the paper
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}
"""

"""
Modified by Wenzhu Xing
"""


parser = argparse.ArgumentParser(description='SISR Unfolding test *^_^*')
parser.add_argument('--model', type=str, required=False, dest='model',
                    default='URADMMNet')
parser.add_argument('--img_path', type=str, required=False, dest='img_path',
                    default='CBSD68')
parser.add_argument('--kmode', type=str, required=False, dest='kmode',
                    default='GM')
parser.add_argument('--reuslt_path', type=str, required=False, dest='reuslt_path',
                    default='results/Unfolding/')
parser.add_argument('--n_channels', type=int, required=False, default=1, dest='n_channels')
parser.add_argument('--iter_num', type=int, required=False, dest='iter_num', default=6) # ADMM_Relax_H3
parser.add_argument('--scale_factor', type=int, required=False, dest='sf', default=1)
parser.add_argument('--kernel_size', type=int, required=False, dest='kernel_size', default=2)
parser.add_argument('--kernel_mode', type=str, required=False, dest='kernel_mode', default='gau')
parser.add_argument('--noise_level', type=float, required=False, dest='noise_level', default=0.0)
parser.add_argument('--show_img', required=False, action='store_true', default=False, dest='show_img')
parser.add_argument('--gpu', required=False, action='store_true', default=False, dest='use_gpu')
parser.add_argument('--device', type=str, required=False, default='cuda:0')
config = parser.parse_args()

# ----------------------------------------
# Preparation
# ----------------------------------------
model_name = 'model_' + config.model + '_18000'      # 'usrgan' | 'usrnet' | 'usrgan_tiny' | 'usrnet_tiny'
testset_name = 'data/'      # test set,  'set5' | 'srbsd68'
test_sf = [2,3] #[2, 3] if config.noise_level == 7.65 else 

show_img = False           # default: False
save_L = True              # save LR image
save_E = True              # save estimated image
save_LEH = False           # save zoomed LR, E and H images

# ----------------------------------------
# load testing kernels
# ----------------------------------------
# kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernels.mat'))['kernels']
kernels = loadmat('kernels_12.mat')['kernels']

n_channels = config.n_channels # if 'gray' in  model_name else 3  # 3 for color image, 1 for grayscale image
model_pool = 'model_zoo'  # fixed
results = config.reuslt_path     # fixed
noise_level_img = config.noise_level/255.0    # fixed: 0, noise level for LR image
noise_level_model = noise_level_img  # fixed, noise level of model, default 0
result_name = config.img_path + '_' + config.model+'_SISR_gray_'+config.kmode + '_n' + str(config.noise_level)
model_path = os.path.join(model_pool, model_name)#+'.pth')

# ----------------------------------------
# L_path = H_path, E_path, logger
# ----------------------------------------
L_path = os.path.join(testset_name, config.img_path)  # L_path and H_path, fixed, for Low-quality images
E_path = os.path.join(results, result_name)    # E_path, fixed, for Estimated images
util.mkdir(E_path)

logger_name = result_name
utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
logger = logging.getLogger(logger_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------
# load model
# ----------------------------------------
if 'ADMM' in model_name:
    from model.network_usrnet_v1 import URADMMNet as net
else:
    from model.network_usrnet_v1 import USRNet as net
model = net(n_iter=config.iter_num, h_nc=32, in_nc=n_channels+1, out_nc=n_channels, nc=[16, 32, 64, 64],
            nb=2, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
    
model = DataParallel(model)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for key, v in model.named_parameters():
    v.requires_grad = False
number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
model = model.to(device)

logger.info('Model path: {:s}'.format(model_path))
logger.info('Params number: {}'.format(number_parameters))
logger.info('Model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
logger.info(L_path)
L_paths = util.get_image_paths(L_path)

# --------------------------------
# read images
# --------------------------------
test_results_ave = OrderedDict()
test_results_ave['psnr_sf_k'] = []
test_results_ave['psnr_sf'] = []

if config.kmode == 'GM':
    klist = [2,3,5,7,8,10]
elif config.kmode == 'GK':
    klist = [2,3,5,7]
elif config.kmode == 'MK':
    klist = [8,9,10,11]
elif config.kmode == 'RK':
    kernels = hdf5storage.loadmat('Levin09.mat')['kernels']
    klist = [0,1,2,3]

for sf in test_sf:
    psnr_k = 0.0

    for k_index in klist: #kernels.shape[1]

        test_results = OrderedDict()
        test_results['psnr'] = []
        kernel = kernels[0, k_index].astype(np.float64)

        ## other kernels
        # kernel = utils_deblur.blurkernel_synthesis(h=25)  # motion kernel
        # kernel = utils_deblur.fspecial('gaussian', 25, 1.6) # Gaussian kernel
        # kernel = sr.shift_pixel(kernel, sf)  # pixel shift; optional
        # kernel /= np.sum(kernel)

        util.surf(kernel) if show_img else None
        idx = 0

        for img in L_paths:

            # --------------------------------
            # (1) classical degradation, img_L
            # --------------------------------
            idx += 1
            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)  # HR image, int8
            img_H = util.modcrop(img_H, np.lcm(sf,8))  # modcrop
            # img_H = util.modcrop(img_H, sf)  # modcrop
            # img_H = util.modcrop(img_H, 16) 
            
            # generate degraded LR image
            img_L = ndimage.filters.convolve(img_H, kernel[..., np.newaxis], mode='wrap')  # blur
            img_L = sr.downsample_np(img_L, sf, center=False)  # downsample, standard s-fold downsampler
            img_L = util.uint2single(img_L)  # uint2single

            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img, img_L.shape) # add AWGN

            util.imshow(util.single2uint(img_L)) if show_img else None

            x = util.single2tensor4(img_L)
            k = util.single2tensor4(kernel[..., np.newaxis])
            sigma = torch.tensor(noise_level_model).float().view([1, 1, 1, 1]) 
            [x, k, sigma] = [el.to(device) for el in [x, k, sigma]]

            # --------------------------------
            # (2) inference
            # --------------------------------
            x = model(x, k, sf, sigma)

            # --------------------------------
            # (3) img_E
            # --------------------------------
            img_E = util.tensor2uint(x)
            # img_E = img_E[:,:,0]
            
            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index+1)+'_'+model_name+'.png'))


            # --------------------------------
            # (4) img_LEH
            # --------------------------------
            img_L = util.single2uint(img_L)
            if save_LEH:
                k_v = kernel/np.max(kernel)*1.2
                k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                k_v = cv2.resize(k_v, (3*k_v.shape[1], 3*k_v.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I = cv2.resize(img_L, (sf*img_L.shape[1], sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I[:k_v.shape[0], -k_v.shape[1]:, :] = k_v
                img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                util.imshow(np.concatenate([img_I, img_E, img_H], axis=1), title='LR / Recovered / Ground-truth') if show_img else None
                util.imsave(np.concatenate([img_I, img_E, img_H], axis=1), os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index+1)+'_LEH.png'))

            if save_L:
                util.imsave(img_L, os.path.join(E_path, img_name+'_x'+str(sf)+'_k'+str(k_index+1)+'_LR.png'))

            psnr = util.calculate_psnr(img_E, img_H[:,:,0], border=sf**2)  # change with your own border
            test_results['psnr'].append(psnr)
            logger.info('{:->4d}--> {:>10s} -- x{:>2d} --k{:>2d} PSNR: {:.2f}dB'.format(idx, img_name+ext, sf, k_index, psnr))

        ave_psnr_k = sum(test_results['psnr']) / len(test_results['psnr'])
        psnr_k = psnr_k + ave_psnr_k
        logger.info('------> Average PSNR(RGB) of ({}) scale factor: ({}), kernel: ({}) sigma: ({}): {:.2f} dB'.format(testset_name, sf, k_index+1, noise_level_model, ave_psnr_k))
        test_results_ave['psnr_sf_k'].append(ave_psnr_k)
    logger.info('------> Average PSNR(RGB) of ({}) scale factor: ({}), average all kernels, sigma: ({}): {:.2f} dB'.format(testset_name, sf, noise_level_model, psnr_k/len(klist)))
    test_results_ave['psnr_sf'].append(psnr_k/len(klist))
logger.info(test_results_ave['psnr_sf_k'])
logger.info(test_results_ave['psnr_sf'])


# if __name__ == '__main__':

#     main()
