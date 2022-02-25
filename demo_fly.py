# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:03:58 2022

@author: Amin
"""

import glob
import torch
import numpy as np
import models as M
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# %%
device = 'cuda'
bs = 8
folder = '..\\data\\Fly Wings\\20X_magnification_Leica\\20X_magnification\\'

# %%
import pandas as pd
leica = pd.read_csv('..\\data\\Fly Wings\\Leica_2X_coords.tsv',sep='\t')
pc = leica.to_numpy()[:,8:-1].reshape(48,len(leica),2)
for i in range(100):
    plt.scatter(pc[:10,i,0],pc[:10,i,1],c='r')

olympus = pd.read_csv('..\\data\\Fly Wings\\Olympus_2X_coords.tsv',sep='\t')
pc = olympus.to_numpy()[:,8:-1].reshape(48,len(olympus),2)
for i in range(100):
    plt.scatter(pc[:10,i,0],pc[:10,i,1],c='b')
# %%
from PIL import Image

files = glob.glob(folder+'samw_F_L_lei_2X*')[:32]
imgs = [np.array(Image.open(file))[100:-100:4,100:-100:4,None,:].astype(float)/255 for file in files]

dataset = D.ImageDataset(np.array(imgs[1:]).transpose([0,4,1,2,3]),device=device)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

plt.imshow(imgs[0].squeeze())
# %%

A = torch.tensor(imgs[0].transpose([3,0,1,2])).to(device).float()

model = M.PiecewiseRigid(sz=A.shape[1:],A=A,positions=None,centers=None,n_channels=3,device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

losses = M.train_model(model,dataloader,optimizer,gamma=0,epochs=40,epochs_theta=200,device=device)

# %%
plt.figure()
plt.plot(losses[0:])
plt.xlabel('Iterations')
plt.ylabel('Loss')
# %%
testloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
registered = np.zeros(list(A.shape[1:])+[3,len(dataset)])
for batch_idx, data in enumerate(testloader):
    x_t,flow,_,_ = model(data[0].to(device),data[1])
    x_t = x_t.detach().cpu().numpy()
    registered[:,:,:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(x_t,[2,3,4,1,0])

# %%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.array(imgs[1:]).mean(0).mean(2))
plt.title('Before')
plt.subplot(1,2,2)
plt.imshow(registered.mean(4).mean(2))
plt.title('After')

# %%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(imgs[6].squeeze())
plt.title('Before')
plt.subplot(1,2,2)
plt.imshow(registered[:,:,0,:,5])
plt.title('After')
