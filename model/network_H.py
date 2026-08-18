#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:32:59 2024

"""

import torch.nn as nn
import torch

class HyPaNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=10, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.clamp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, 1, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        output = self.mlp(x) + 1e-6
        alpha = self.clamp(x)
        return torch.cat((output, alpha), dim = 1)

class Allin1_HyPaNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=10, channel=64, alpha=True):
        super(Allin1_HyPaNet, self).__init__()
        if alpha:
            self.iter = out_nc*2
        else:
            self.iter = out_nc
        self.mlp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.clamp = nn.Sequential(
                nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, self.iter, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, sf, n):
        x = torch.cat((n, sf), dim = 1)
        mu = self.mlp(x) + 1e-6
        n_alpha = self.clamp(x)
        
        return torch.cat((mu, n_alpha), dim = 1)       