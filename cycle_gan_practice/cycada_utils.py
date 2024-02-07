#!/usr/bin/env python
# coding: utf-8

# In[2]:

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import random
import torch
import torch.nn as nn
import config
import copy
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
import os
import torchvision.models as models
import torch.nn.functional as F
from torchvision import datasets, models, transforms


PROJECT_ROOT_DIR = "."
PROJECT_SAVE_DIR = "Figure_PDFs"
if not (os.path.isdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)):
    print('Figure directory didn''t exist, creating now.')
    os.mkdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)
else:
    print('Figure directory exists.') 
    
img_height   = 256
img_width    = 256
channels     = 3

transforms_ = [
    transforms.Resize(int(img_height*1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

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
            
            
            
class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#         sequence += [nn.Softmax(dim=1)] 
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class LeNet(nn.Module):
    def __init__(self, input_nc):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(input_nc, 20, kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.dropout2 = nn.Dropout2d(0.5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU(True)

        self.flatten = Flatten()

        self.fc1 = nn.Linear(186050, 500)
        self.relu3 = nn.ReLU(True)
        self.dropout3 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(500, 2)

    def forward(self, inp, d_feat=False):
        """Standard forward."""
        out = self.conv1(inp)
        out = self.maxpool1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.maxpool2(out)
        out = self.relu2(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu3(out)
        out = self.dropout3(out)

        out = self.fc2(out)

        if d_feat:
            return out
        else:
            return out

class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        
        
        for param in resnet50.parameters():
            param.requires_grad = False
        
        # Model initialization
        self.resnet_pretrained = resnet50                  #Running images through CNN
        self.fc1 = nn.Linear(self.resnet_pretrained.fc.out_features, 50)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)        
    def forward(self, image):
        img_features = self.resnet_pretrained(image)
        img_features = torch.flatten(img_features, 1)
        img_features = self.fc1(img_features)
        x = self.relu(img_features)
        x = self.dropout(x)
        x = self.fc2(x.float())
        return x
    
    
#input is 2 beacuse I have 2 cities currently in my source domain?
class FeatureDiscriminator(nn.Module):
    def __init__(self):
        super(FeatureDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(2, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
        )

    def forward(self, score):
        out = self.discriminator(score)
        return out
    
    
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            target_real_label (float) - label for a real image
            target_fake_label (float) - label of a fake image
        Note: Do not use sigmoid as the last layer of the Discriminator.
        LSGAN needs no sigmoid. Vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - typically the prediction from a discriminator
            target_is_real (bool) - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth labels, with the same size as the input
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given the Discriminator's output and ground truth labels.
        Parameters:
            prediction (tensor) - typically the prediction output from a discriminator
            target_is_real (bool) - if the ground truth label is for real images or fake images
        Returns:
            - The calculated loss.
            - The discriminator accuracy (ratio of correct predictions to total predictions).
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)

        # Calculate discriminator accuracy
        pred_labels = torch.round(torch.sigmoid(prediction)).squeeze()
        true_labels = torch.ones_like(pred_labels) if target_is_real else torch.zeros_like(pred_labels)
        
        correct_predictions = torch.sum(pred_labels == true_labels).item()
        total_predictions = true_labels.size(0)  # Use the size() method to get the tensor size
        
        accuracy = correct_predictions / total_predictions
        
        return loss, accuracy
    
def sample_images(dataloader, e, i):
    """show a generated sample from the test set"""
    imgs = next(iter(dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = imgs['A'].type(Tensor) # A : monet
    fake_B = G_AB(real_A).detach()
    real_B = imgs['B'].type(Tensor) # B : photo
    fake_A = G_BA(real_B).detach()
    recona = G_BA(fake_B).detach()
    reconb = G_AB(fake_A).detach() 

    # Resize images to 10 by 10
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True, scale_each=True, padding=1)
    fake_B = make_grid(fake_B, nrow=5, normalize=True, scale_each=True, padding=1)
    real_B = make_grid(real_B, nrow=5, normalize=True, scale_each=True, padding=1)
    fake_A = make_grid(fake_A, nrow=5, normalize=True, scale_each=True, padding=1)
    reconA = make_grid(recona, nrow=5, normalize=True, scale_each=True, padding=1)
    reconB = make_grid(reconb, nrow=5, normalize=True, scale_each=True, padding=1)

    # Arange images along y-axis    
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A, reconA, reconB), 1)
    plt.imshow(image_grid.cpu().permute(1,2,0))
    plt.title('Real A vs Fake B | Real B vs Fake A| Recon A vs Recon B')
    plt.axis('off')
    plt.gcf().set_size_inches(10, 10) # set image size to 10 by 10 inches
    plt.savefig(os.path.join('Figure_PDFs', f'epoch_{str(e)}_iter{str(i)}.png' ))
    plt.show();
    
def discriminator_acc(prediction, target_is_real=True):
    pred_labels = torch.round(torch.sigmoid(prediction)).squeeze()
    true_labels = torch.ones_like(pred_labels) if target_is_real else torch.zeros_like(pred_labels)

    correct_predictions = torch.sum(pred_labels == true_labels).item()
    total_predictions = true_labels.size(0) 

    accuracy = correct_predictions / total_predictions

    return accuracy