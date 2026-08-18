# -*- coding: utf-8 -*-
import torch.fft
import torch
from scipy import ndimage
import numpy as np
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator as RGI
import random


def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


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


def upsample(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]



def data_solution(x, FB, FBC, F2B, FBFy, mu, sf):
    FR = FBFy + torch.fft.fftn(mu*x, dim=(-2,-1))
    x1 = FB.mul(FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = FBR.div(invW + mu)
    FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
    FX = (FR-FCBinvWBR)/mu
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

    return Xest

def data_solution2(z, xb, FB, FBC, F2B, FBFy, lamda, mu, sf):
    Phi = mu + lamda*F2B
    FR = FBFy + torch.fft.fftn(mu*z, dim=(-2,-1)) + FBC*torch.fft.fftn(lamda*xb, dim=(-2,-1))
    x1 = FB.mul(FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B+lamda*F2B, sf), dim=-1, keepdim=False)
    invWBR = FBR.div(invW + mu)
    FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
    FX = (FR-FCBinvWBR)/Phi
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

    return Xest

def deblur_solution(xb, z, FBC, F2B, lamda):
    FXB = torch.fft.fftn(xb, dim=(-2,-1))
    FR = FBC.mul(FXB) + torch.fft.fftn(lamda*z, dim=(-2,-1))
    FX = FR.div(F2B + lamda)
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

    return Xest

def interpolation(STy, x_tilde, r, mu):
    """
    y_lr     : (H/r, W/r) 低分辨率观测
    x_tilde  : (H, W)     先验估计（如 denoiser 输出）
    r        : 下采样因子 (integer)
    mu      : ADMM 参数
    
    返回：
    x_hat    : (H, W) 的数据一致性闭式解
    """

    device = x_tilde.device
    H, W = x_tilde.shape[-2:]

    # 1) 构造 mask S^T S (H x W)
    mask = torch.zeros(x_tilde.shape, device=device, dtype=x_tilde.dtype)
    mask[..., 0:H:r, 0:W:r] = 1.0  # 被采样的位置 = 1

    # 3) 逐元素闭式 update
    y_hat = (STy + mu * x_tilde) / (mask + mu)

    return y_hat

def interpolation2(STy, Hx, xb_tilde, r, mu1, mu2):
    """
    y_lr     : (H/r, W/r) 低分辨率观测
    x_tilde  : (H, W)     先验估计（如 denoiser 输出）
    r        : 下采样因子 (integer)
    mu      : ADMM 参数
    
    返回：
    x_hat    : (H, W) 的数据一致性闭式解
    """

    device = xb_tilde.device
    H, W = xb_tilde.shape[-2:]

    # 1) 构造 mask S^T S (H x W)
    mask = torch.zeros(xb_tilde.shape, device=device, dtype=xb_tilde.dtype)
    mask[..., 0:H:r, 0:W:r] = 1.0  # 被采样的位置 = 1

    # 3) 逐元素闭式 update
    y_hat = (STy + mu1 * Hx + mu2 * xb_tilde) / (mask + mu1 + mu2)

    return y_hat

def data_SRDB(x, FB, FBC, F2B, FBFy, mu, sf, alpha=1.0):
    # x: z - u
    # Interpolation first
    # y_hat = alpha*interpolation(STy, x, sf, mu)+(1-alpha)*x
    # # nnon-blind Deblur
    # FBFy = FBC*torch.fft.fftn(y_hat, dim=(-2, -1))
    # FR = FBFy + mu*torch.fft.fftn(x, dim=(-2,-1))
    # FXest = FR.div(F2B + mu)
    # Xest = torch.real(torch.fft.ifftn(FXest, dim=(-2, -1)))
    FR = FBFy + torch.fft.fftn(mu*x, dim=(-2,-1))
    x1 = FB.mul(alpha*FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    x2 = FB.mul((1-alpha)*torch.fft.fftn(x, dim=(-2,-1)))
    FBX = torch.mean(splits(x2, sf), dim=-1, keepdim=False)
    invWBR = FBR.div(invW + mu)+FBX
    FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
    FX = (FR-FCBinvWBR)/mu
    Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))
    
    return Xest
    
def pre_calculate2(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FB = p2o(k, (w*sf, h*sf))
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample(x, sf=sf)
    # FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
    return FBC, F2B, STy

def pre_calculate(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    # b, c, w, h = x.shape#[-2:]
    w, h = x.shape[-2:]
    FB = p2o(k, (w*sf, h*sf))
    # FB = torch.ones(b, c, w*sf, h*sf).type_as(x)
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample(x, sf=sf)
    FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy

def classical_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    #x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH, image or kernel
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x

def shift_pixel2(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH, image or kernel
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)
    x11, y11 = np.meshgrid(x1, y1, indexing='ij', sparse=True)

    if x.ndim == 2:
        r = RGI((xv, yv), x.T, method='linear')
        x_t = r((x11, y11))
        x = x_t.T
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            r = RGI((xv, yv), x[:, :, i].T, method='linear')
            x_t = r((x11, y11))
            x[:, :, i] = x_t.T

    return x

def gen_kernel(k_size=np.array([25, 25]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=12., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    sf = random.choice([1, 2, 3, 4])
    scale_factor = np.array([sf, sf])
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = 0#-noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5*(scale_factor - 1) # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    #kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel

def pad_circular(input, padding):
    # type: (Tensor, List[int]) -> Tensor
    """
    Arguments
    :param input: tensor of shape :math:`(N, C_{\text{in}}, H, [W, D]))`
    :param padding: (tuple): m-elem tuple where m is the degree of convolution
    Returns
    :return: tensor of shape :math:`(N, C_{\text{in}}, [D + 2 * padding[0],
                                     H + 2 * padding[1]], W + 2 * padding[2]))`
    """
    offset = 3
    for dimension in range(input.dim() - offset + 1):
        input = dim_pad_circular(input, padding[dimension], dimension + offset)
    return input


def dim_pad_circular(input, padding, dimension):
    # type: (Tensor, int, int) -> Tensor
    input = torch.cat([input, input[[slice(None)] * (dimension - 1) +
                      [slice(0, padding)]]], dim=dimension - 1)
    input = torch.cat([input[[slice(None)] * (dimension - 1) +
                      [slice(-2 * padding, -padding)]], input], dim=dimension - 1)
    return input

def imfilter(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    '''
    x = pad_circular(x, padding=((k.shape[-2]-1)//2, (k.shape[-1]-1)//2))
    x = torch.nn.functional.conv2d(x, k, groups=x.shape[1])
    return x





