#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from __future__ import print_function, division


import numpy as np
np.random.seed(42)
import scipy
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import math
import pandas as pd
from datetime import datetime, timedelta
import os
from os import listdir
from os.path import isfile, join
import glob
import pickle as pkl
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torchvision.models as models
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset,ConcatDataset
import time
from joblib import dump, load
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomTreesEmbedding
from tqdm import tqdm 
import itertools
import torch.nn.functional as F
import torch.utils.data as torchdata


# Where to save the figures
PROJECT_ROOT_DIR = "."
PROJECT_SAVE_DIR = "Figure_PDFs"

import os
if not (os.path.isdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)):
    print('Figure directory didn''t exist, creating now.')
    os.mkdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)
else:
    print('Figure directory exists.') 

# Define a function to save future figures to PDFs

import statistics


# In[1]:


from scipy import stats
from sklearn import metrics

def savepdf(fig,name):
    fig.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+name+'.png')

# To evaluate the statistics between predicted and true PM2.5
def eval_stat(y_train_pred, y_train):
#     y_train_pred = y_train_pred.detach().numpy()
    
    Rsquared = stats.spearmanr(y_train_pred, y_train)[0]
    pvalue = stats.spearmanr(y_train_pred, y_train)[1]
    Rsquared_pearson = stats.pearsonr(np.squeeze(y_train_pred), y_train)[0]
    pvalue_pearson = stats.pearsonr(np.squeeze(y_train_pred), y_train)[1]
    return Rsquared, pvalue, Rsquared_pearson, pvalue_pearson


def plot_result(prediction, y_true, Rsquared, pvalue, Rsquared_pearson, pvalue_pearson, plot_label="train", lower_bound=0, upper_bound=0,save = True, fig_name = ''):
    fig, ax = plt.subplots(figsize = (10,10))
    if plot_label == "train":
        ax.scatter(y_true, prediction, color = 'purple', alpha=0.5, edgecolors=(0, 0, 0),  s = 100)
        ax.text(0.99, 0.99, 'Training Dataset', ha = 'right', va='top', color='purple', weight='bold', fontsize=16,                     transform=ax.transAxes)
    else:
        ax.scatter(y_true, prediction, color = 'green', alpha=0.5, edgecolors=(0, 0, 0),  s = 100)
        ax.text(0.99, 0.99, 'Testing Dataset', ha = 'right', va='top', color='green', weight='bold', fontsize=16,                   transform=ax.transAxes)
        
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], 'k--', lw=1)
    ax.set_xlabel('True $PM_{2.5}$ ($\mu $g m$^{-3}$)', size = 20)
    ax.set_ylabel('Predicted $PM_{2.5}$ ($\mu $g m$^{-3}$)', size = 20)
    ax.text(0.05, 0.9, 'Spearman r: '+ str(round(Rsquared, 2)), bbox=dict(facecolor="black", alpha=0.1), ha='left', va='top',       color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.05, 0.80, 'Spearman p-value: '+ str(round(pvalue,2)), bbox=dict(facecolor='black', alpha = 0.1),ha='left',             va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes) 
    ax.text(0.05, 0.70, 'Pearson r: '+ str(round(Rsquared_pearson,2)), bbox=dict(facecolor='black', alpha = 0.1),ha='left',         va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.05, 0.60, 'Pearson p-value: '+ str(round(pvalue_pearson,2)),bbox=dict(facecolor='black', alpha = 0.1),ha='left',       va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.text(0.05, 0.50, 'RMSE: '+ str(round(np.sqrt(metrics.mean_squared_error(y_true, prediction)),2)),                             bbox=dict(facecolor='black', alpha = 0.1),ha='left', va='top', color='black', weight='roman', fontsize=16,                       transform=ax.transAxes)
    ax.text(0.05, 0.40, 'MAE: '+ str(round(metrics.mean_absolute_error(y_true, prediction),2)), bbox=dict(facecolor='black',         alpha = 0.1), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
#     ax.text(0.05, 0.65, '% error: '+ str(round(metrics.mean_absolute_error(y_true, prediction)/np.mean(y_true)*100,1))+'%', bbox=dict(facecolor='black', alpha = 0.2), ha='left', va='top', color='black', weight='roman', fontsize=16, transform=ax.transAxes)
    ax.axis('tight')
    ax.tick_params(labelsize = 16)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
#     plt.savefig(PROJECT_ROOT_DIR+'/'+'predictions'+'/'+(str(lab1)+'.png', facecolor='w')
#     if save:
#         savepdf(fig,fig_name)
    ax.figure.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+fig_name+'.png')
    plt.close(fig)
    return


#for plotting results
def train_test_loss(loss_train, loss_test, epochs, save = True, fig_name=''):
    fig, ax = plt.subplots(figsize = (7,5))
    PROJECT_ROOT_DIR = "."
    PROJECT_SAVE_DIR = "Figure_PDFs"
    epoch = range(epochs)
    ax.plot(epoch, loss_train, color='g', linewidth=0.5, label='Train loss')
    ax.plot(epoch, loss_test, color='r', linewidth=0.5, label='Test loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Train and test loss')
    ax.legend()
    plt.show()
    if save:
        fig.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+fig_name+'.png')

def detach_predictions(predictions):
    predictions_array = predictions.cpu().detach()
    predictions_array = np.array(predictions_array)
    return predictions_array

def retrieve_true_PM(trues):
    true_PM = []
    for i in range(len(trues)):
        np.array(true_PM.append(trues[i][1]))
    return true_PM
def Gradient_Penalty(mixed_scores, mixed_input):

    gradient= torch.autograd.grad(outputs= mixed_scores,
                                    inputs= mixed_input,
                                    grad_outputs= torch.ones_like(mixed_scores),
                                    create_graph=True,
                                    retain_graph=True)[0]
    gradient_norm= gradient.view(len(gradient), -1).norm(2, dim=1)
    penalty = torch.nn.MSELoss()(gradient_norm, torch.ones_like(gradient_norm))*10
    return penalty

def normalize_PM(PM_array):
    normal_PM = []
    pm_mean = np.mean(PM_array)
    pm_std = np.std(PM_array)
    for i in PM_array:
        z = (i-pm_mean)/pm_std
        normal_PM.append(z)
    return normal_PM

class Temp_Dataset(Dataset):
    def __init__(self, file_path, transform):
        self.file_path = file_path
        self.transform = transform
        self.labels = []
        self.imgs = []
        
        with open(file_path,'rb') as fp:
            images = pkl.load(fp)

            for station in images:
                for datapoints in station:
                    self.labels.append(datapoints['PM'])
                    self.imgs.append(datapoints['Image'][:,:,:3])
                                       
    def __len__(self): return len(self.imgs)
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  
            
        label = self.labels[idx]
        image = self.imgs[idx]
        
        if self.transform:
            trans = transforms.Compose([transforms.ToPILImage(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(degrees=42),
                                      transforms.CenterCrop(size=100),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
            image = trans(image)
            
        sample = {'img': image,
                  'lbl': label
        }
        
        return sample
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
#regressor task--> needs to take image and output PM (f--task model in cycada paper)
class Regressor_Task(nn.Module):
    def __init__(self):
        super(Regressor_Task, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        
        # Freeze all feature extraction layers in the encoder
        for param in resnet50.parameters():
            param.requires_grad = False
        
        # Model initialization
        self.resnet_pretrained = resnet50                  #Running images through CNN
        self.fc1 = nn.Linear(self.resnet_pretrained.fc.out_features, 50)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)        
    def forward(self, image):
        img_features = self.resnet_pretrained(image)
        img_features = torch.flatten(img_features, 1)
        img_features = self.fc1(img_features)
        x = self.relu(img_features)
        x = self.dropout(x)
        x = self.fc2(x.float())
        return x
class ResnetBlock(torch.nn.Module):
    def __init__(self,num_filter,kernel_size=3,stride=1,padding=0):
        super(ResnetBlock,self).__init__()

        conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size,
                                stride, padding)
        
        conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size,
                                stride, padding)
        
        bn = torch.nn.InstanceNorm2d(num_filter)
        relu = torch.nn.ReLU(True)
        pad = torch.nn.ReflectionPad2d(1)
        
        self.resnet_block = torch.nn.Sequential(
            pad,
            conv1,
            bn,
            relu,
            pad,
            conv2,
            bn
            )
    def forward(self,x):
        out = self.resnet_block(x)
        return out    

class Generator(nn.Module):
    def __init__(self, input_dim, num_filter, num_res):
        super(Generator, self).__init__()
        
        #ENCODE
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=num_filter, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(num_filter),
            nn.ReLU(inplace=True))
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(in_channels = num_filter, out_channels = num_filter*2, kernel_size = 3, stride=2, padding=1),
            nn.InstanceNorm2d(num_filter*2),
            nn.ReLU(inplace=True))
        
        self.encode3 = nn.Sequential(
            nn.Conv2d(num_filter*2, num_filter*4, 3, 2, 1),
            nn.InstanceNorm2d(num_filter*4),
            nn.ReLU(inplace=True))
        
        #TRANSFORM
        self.resnet_block = []
        for n in range(num_res):
            self.resnet_block.append(ResnetBlock(num_filter*4))
            self.resnet_block = nn.Sequential(*self.resnet_block)
            
        #DECODE    
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_filter*4, out_channels=num_filter*2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_filter*2),
            nn.ReLU(inplace=True))
        
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_filter*2, out_channels=num_filter, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(num_filter),
            nn.ReLU(inplace=True))
            
        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=num_filter, out_channels=input_dim, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(num_filter),
            nn.Tanh())
        
    def forward(self,img):
        enc1 = self.encode1(img)
        enc2 = self.encode2(enc1)
        enc3 = self.encode3(enc2)
        
        res = self.resnet_block(enc3)
        
        dec1 = self.decode1(res)
        dec2 = self.decode2(dec1)
        out = self.decode3(dec2)
        return out
    
    
class Discriminator(nn.Module):
    def __init__(self, input_dim, num_filter):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=num_filter, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=num_filter, out_channels=num_filter*2, kernel_size=4, stride=2, padding=2),
            nn.InstanceNorm2d(num_filter*2),
            nn.LeakyReLU(0.2))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_filter*2, out_channels=num_filter*4, kernel_size=4, stride=2, padding=2),
            nn.InstanceNorm2d(num_filter*4),
            nn.LeakyReLU(0.2))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=num_filter*4, out_channels=num_filter*8, kernel_size=4, stride=2, padding=2),
            nn.InstanceNorm2d(num_filter*8),
            nn.LeakyReLU(0.2))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=num_filter*8, out_channels=num_filter*8, kernel_size=4, stride=1, padding=2),
            nn.InstanceNorm2d(num_filter*8),
            nn.LeakyReLU(0.2))
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=num_filter*8, out_channels=1, kernel_size=4, stride=1, padding=2))
        
        self.linear = nn.Linear(1 * 100, 1)  # Add a linear layer with output channel of 1 If the size 
        
    def forward(self, img):
        lay1 = self.conv1(img)
        lay2 = self.conv2(lay1)
        lay3 = self.conv3(lay2)
        lay4 = self.conv4(lay3)
        lay5 = self.conv5(lay4)
        out = self.conv6(lay5)
#         print(out.shape)
        out = out.view(out.size(0), -1)  # Flatten the output tensor
#         print(out.shape)
        out = self.linear(out)  # Apply the linear layer to the flattened output tensor
#         print(out.shape)
        out = nn.Sigmoid()(out) 
        return out

# class Discriminator(nn.Module):
#     def __init__(self, input_dim, num_filter):
#         super(Discriminator, self).__init__()
        
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=input_dim, out_channels=num_filter, kernel_size=4, stride=2, padding=2),
#             nn.LeakyReLU(0.2))
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=num_filter, out_channels=num_filter*2, kernel_size=4, stride=2, padding=2),
#             nn.InstanceNorm2d(num_filter*2),
#             nn.LeakyReLU(0.2))
        
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=num_filter*2, out_channels=num_filter*4, kernel_size=4, stride=2, padding=2),
#             nn.InstanceNorm2d(num_filter*4),
#             nn.LeakyReLU(0.2))
        
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=num_filter*4, out_channels=num_filter*8, kernel_size=4, stride=2, padding=2),
#             nn.InstanceNorm2d(num_filter*8),
#             nn.LeakyReLU(0.2))
        
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(in_channels=num_filter*8, out_channels=num_filter*8, kernel_size=4, stride=1, padding=2),
#             nn.InstanceNorm2d(num_filter*8),
#             nn.LeakyReLU(0.2))
        
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(in_channels=num_filter*8, out_channels=1, kernel_size=4, stride=1, padding=2),
#             nn.Softmax(dim=1))
        
#     def forward(self, img):
#         lay1 = self.conv1(img)
#         lay2 = self.conv2(lay1)
#         lay3 = self.conv3(lay2)
#         lay4 = self.conv4(lay3)
#         lay5 = self.conv5(lay4)
#         out = self.conv6(lay5)
#         return out
    
    
class Discriminator_Task(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        
        super(Discriminator_Task,self).__init__()
        
        
        self.lin1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                            nn.ReLU())
        self.lin2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU())
        self.linout = nn.Sequential(nn.Linear(hidden_dim, 1),
                              nn.Sigmoid())
        
    def forward(self, score):
        lin1 = self.lin1(score)
        lin2 = self.lin2(lin1)
        out = self.linout(lin2)
        return torch.round(out)   
    
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        
        self.loss = nn.BCEWithLogitsLoss()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
    def get_target_tensor(self,prediction, target_is_real):
        
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
            
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        
        target_tensor=self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        
        return loss    