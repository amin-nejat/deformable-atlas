# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:35:46 2022

@author: Amin
"""
from torch.utils.data import DataLoader
import torch.optim as optim
import visualization as V
import pydicom as dicom
import datasets as D
import models as M
import numpy as np
import torch
import glob

# %%
device = 'cuda'
bs = 2
folder = '..\\data\\Rhesus Macaque'

# %% Load image data from subjects
imgs_ = []
for folder in glob.glob(folder+'\\*'):
    files = glob.glob(folder+'\\*.dcm')
    img = np.zeros((256,256,82))
    
    for i,file in enumerate(files):
        ds = dicom.dcmread(file)
        if i < 82:
            img[:,:,i] = ds.pixel_array
    
    imgs_.append(img)
# %% Visualize subjects

V.visualize_subjects(imgs_)
V.multi_slice_viewer(imgs_[1].transpose(2,0,1))

# %% Create a dataset for training
imgs = [img[:,:,::10,None][:,:,:,[0,0,0,0]] for img in imgs_]
dataset = D.ImageDataset(np.array(imgs[1:]).transpose([0,4,1,2,3]),device=device)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

# %% Initialize atlas and train 
A = torch.tensor(imgs[0].transpose([3,0,1,2])).to(device).float()
model = M.ImagePiecewiseRigid(sz=A.shape[1:],A=A,positions=None,centers=None,device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
losses = M.train_model(model,dataloader,optimizer,gamma=1e-3,epochs=100,device=device)

# %%
V.plot_loss(losses[30:])

# %%
testloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
registered = np.zeros(list(A.shape[1:])+[4,len(dataset)])
for batch_idx, data in enumerate(testloader):
    x_t,flow,_,_ = model(data[0].to(device),data[1])
    x_t = x_t.detach().cpu().numpy()
    registered[:,:,:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(x_t,[2,3,4,1,0])

# %%
V.visualize_subjects([np.array(imgs[1:]).mean(0)[:,:,:,0],registered.mean(4)[:,:,:,0]])
