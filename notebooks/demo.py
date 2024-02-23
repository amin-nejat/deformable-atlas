# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:41:57 2022

@author: Amin
"""

# %%
import sys
sys.path.insert(1,'..')

import torch

import visualization
import datasets
import models
import utils

import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
# %%
device = 'cuda' # cuda vs. cpu
n_sample = 10 # number of synthetic worm point clouds sampled
bs = 16 # batch size for training the model

gamma_re=1 # reconstruction weight in the loss
gamma_ss=0 # semi-supervision weight in the loss
gamma_mm=0 # motion regularization weight in the loss
epochs=20 # number of training epochs
epochs_theta=10 # number of training epochs before each update of atlas

# %% Loading the worm atlas and sampling points from it by rotations
point_cloud, tess = utils.simulate_worm_pc(
    'data/atlas_xx_rgb',
    'data/male_data',
    n_sample
)

# Creating a dataset object
dataset = datasets.PCDataset(
    point_cloud[1:,:,:],
    device=device
)

# Creating a loader instance for the dataset
dataloader = DataLoader(
    dataset,
    batch_size=bs,
    shuffle=False, 
    num_workers=0
)

# Tesselation map used for the piecewise rigid motion model 
mask = 1-np.eye(len(tess))
# %% Train the model
sz = torch.tensor([point_cloud[:,i,:].max() for i in range(3)])
model = models.PCPiecewiseRigid(
    A=point_cloud[0], # Initial atlas (can be any sample)
    sz=sz, # Bounds (size) of the image containing point clouds
    tess=tess, # Tessalation map defining the motion model
    mask=mask, # Distance matrix defining what tesselation parts are connected
    device=device # Device to run the model on
)

optimizer = optim.Adam(
    model.parameters(), 
    lr=1e-4 # Learning rate
)
models.train_model(
    model,
    dataloader,
    optimizer,
    gamma_re=gamma_re,
    gamma_ss=gamma_ss,
    gamma_mm=gamma_mm,
    epochs=epochs,
    epochs_theta=epochs_theta,
    device=device
)

# %% Testing the resulting atlas
testloader = DataLoader(
    dataset, 
    batch_size=bs, 
    shuffle=False, 
    num_workers=0
)

# Use atlas's registration model to move the test point clouds to the alinged space
pos_moved = np.zeros((len(dataset),6,point_cloud.shape[2]))
for batch_idx, data in enumerate(testloader):
    x_t,_,_,_ = model(data[0].to(device),data[1])
    pos_moved[batch_idx*bs:batch_idx*bs+x_t.shape[0],:,:] = x_t.detach().cpu().numpy()
    
# %% Visualization of the aligned point cloud and statistical atlas
visualization.visualize_pc(pos_moved,atlas=model.A.T.detach().cpu().numpy(),title_str='Inferred Atlas')
visualization.visualize_pc(point_cloud,title_str='Pre Registration')
