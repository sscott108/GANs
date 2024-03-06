#!/usr/bin/env python
# coding: utf-8

# In[2]:

import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import random
import torch
import torch.nn as nn
import config
import copy
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import rasterio
from rasterio.windows import Window
import numpy as np
from os import listdir
from os.path import isfile, join
np.random.seed(42)
import copy
import matplotlib
import matplotlib.pyplot as plt
import pickle as pkl
import cv2
from PIL import Image, ImageEnhance
import itertools
import torchvision.transforms as transforms
import glob
from tqdm.notebook import tqdm
import warnings
from torchvision.utils import make_grid
import functools

# Where to save the figures
PROJECT_ROOT_DIR = "."
PROJECT_SAVE_DIR = "Figure_PDFs"

import os
if not (os.path.isdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)):
    print('Figure directory didn''t exist, creating now.')
    os.mkdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)
else:
    print('Figure directory exists.') 
   

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) # reset Conv2d's weight(tensor) with Gaussian Distribution
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0) # reset Conv2d's bias(tensor) with Constant(0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # reset BatchNorm2d's weight(tensor) with Gaussian Distribution
            torch.nn.init.constant_(m.bias.data, 0.0) # reset BatchNorm2d's bias(tensor) with Constant(0)
            
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode
        if self.mode == 'train':
            self.files_A = sorted(glob.glob(os.path.join(root+'/trainA')+'/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root+'/trainB')+'/*.*'))
        elif self.mode == 'test':
            self.files_A = sorted(glob.glob(os.path.join(root+'/testA')+'/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root+'/testB')+'/*.*'))

    def  __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        
        if self.unaligned:
            image_B = Image.open(self.files_B[np.random.randint(0, len(self.files_B)-1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
        if image_A.mode != 'RGB':
            image_A = to_rgb(image_A)
        if image_B.mode != 'RGB':
            image_B = to_rgb(image_B)
            
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {'A':item_A, 'B':item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
#taken directly from cycleGAN Github page.

# class ResnetGenerator(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
#         """Construct a Resnet-based generator
#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#         assert(n_blocks >= 0)
#         super(ResnetGenerator, self).__init__()
        
#         #Instance normalization, applied on each sample and each channel
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
        
        
#         #
#         model = [nn.ReflectionPad2d(3),
#                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]

#         n_downsampling = 2
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
            
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]

#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):       # add ResNet blocks

#             model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                          kernel_size=3, stride=2,
#                                          padding=1, output_padding=1,
#                                          bias=use_bias),
#                       norm_layer(int(ngf * mult / 2)),
#                       nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(3)]
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         model += [nn.Tanh()]

#         self.model = nn.Sequential(*model)

#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input)


# class ResnetBlock(nn.Module):

#     def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         """Initialize the Resnet block
#         A resnet block is a conv block with skip connections
#         We construct a conv block with build_conv_block function,
#         and implement skip connections in <forward> function.
#         Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
#         """
#         super(ResnetBlock, self).__init__()
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
#         """Construct a convolutional block.
#         Parameters:
#             dim (int)           -- the number of channels in the conv layer.
#             padding_type (str)  -- the name of padding layer: reflect | replicate | zero
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers.
#             use_bias (bool)     -- if the conv layer uses bias or not
#         Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
#         """
#         conv_block = []
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)

#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]

#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReflectionPad2d(1)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad2d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#         conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

#         return nn.Sequential(*conv_block)

#     def forward(self, x):
#         """Forward function (with skip connections)"""
#         out = x + self.conv_block(x)  # add skip connections
#         return out
    
# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator"""

#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
#         """Construct a PatchGAN discriminator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]

#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]

#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         self.model = nn.Sequential(*sequence)

#     def forward(self, input):
#         """Standard forward."""
#         return self.model(input)
    
    
    
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):

        super(GANLoss, self).__init__()
        
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.gan_mode = gan_mode
        
        if self.gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
            
        elif self.gan_mode == 'wgangp':
            self.loss = None
            
    def get_target_tensor(self, prediction, target_is_real):
        
        if target_is_real:
            target_tensor = self.real_label
            
        else:
            target_tensor = self.fake_label
        
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        
        if self.gan_mode == 'vanilla':
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
            
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
                
        return loss
    
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad