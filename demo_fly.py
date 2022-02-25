# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 17:03:58 2022

@author: Amin
"""

import glob
import torch
import numpy as np
import models as M
import datasets as D
from PIL import Image
import visualization as V
import torch.optim as optim
from torch.utils.data import DataLoader


# %%
device = 'cuda'
bs = 8
file = '..\\results\\'
folder = '..\\data\\Fly Wings\\20X_magnification_Leica\\20X_magnification\\'
save = False

# %%
sex = 'M'
genotype = 'samw'
side = 'L'

file_name = genotype+'_'+sex+'_'+side

files = glob.glob(folder+file_name+'_lei_2X*')[:32]
imgs = [1-np.array(Image.open(file))[100:-100:4,100:-100:4,None,None,1].astype(float)/255 for file in files]

dataset = D.ImageDataset(np.array(imgs[1:]).transpose([0,4,1,2,3]),device=device)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

V.visualize_subjects(imgs[0:3],titles=[file_name+str(i) for i in range(4)])
# %%

A = torch.tensor(imgs[0].transpose([3,0,1,2])).to(device).float()

model = M.ImagePiecewiseRigid(sz=A.shape[1:],A=A,positions=None,centers=None,n_channels=1,device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

losses = M.train_model(model,dataloader,optimizer,gamma=1e-5,epochs=40,epochs_theta=200,device=device)

# %%
V.plot_loss(losses,save=save,file=file+file_name+'loss')

# %%
testloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
registered = np.zeros(list(A.shape[1:])+[1,len(dataset)])
for batch_idx, data in enumerate(testloader):
    x_t,flow,_,_ = model(data[0].to(device),data[1])
    x_t = x_t.detach().cpu().numpy()
    registered[:,:,:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(x_t,[2,3,4,1,0])

# %%
V.visualize_subjects([np.array(imgs[1:]).mean(0), registered.mean(4)], ['Population','Atlas'],save=save,file=file+file_name+'_atlas')
V.visualize_subjects([imgs[1], registered[:,:,:,:,0]], ['Raw','Registererd'])

