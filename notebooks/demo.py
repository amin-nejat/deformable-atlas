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
device = 'cuda'
n_sample = 10
bs = 16

gamma_re=1
gamma_ss=0
gamma_mm=0
epochs=20
epochs_theta=10

point_cloud, tess = utils.simulate_worm_pc(
    'data/atlas_xx_rgb',
    'data/male_data',
    n_sample
)

dataset = datasets.PCDataset(point_cloud[1:,:,:],device=device)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

mask = 1-np.eye(len(tess))
# %% train
sz = torch.tensor([point_cloud[:,i,:].max() for i in range(3)])
model = models.PCPiecewiseRigid(A=point_cloud[0],sz=sz,tess=tess,mask=mask,device=device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
models.train_model(
    model,dataloader,optimizer,
    gamma_re=gamma_re,gamma_ss=gamma_ss,gamma_mm=gamma_mm,
    epochs=100,epochs_theta=50,device=device
)

# %% test
testloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
pos_moved = np.zeros((len(dataset),6,point_cloud.shape[2]))
for batch_idx, data in enumerate(testloader):
    x_t,_,_,_ = model(data[0].to(device),data[1])
    pos_moved[batch_idx*bs:batch_idx*bs+x_t.shape[0],:,:] = x_t.detach().cpu().numpy()
    
# %% vis
visualization.visualize_pc(pos_moved,atlas=model.A.T.detach().cpu().numpy(),title_str='Inferred Atlas')
visualization.visualize_pc(point_cloud,title_str='Pre Registration')

