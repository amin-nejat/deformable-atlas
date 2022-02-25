# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 18:52:37 2022

@author: Amin
"""

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import visualization as V
import nibabel as nib
import datasets as D
import models as M
import numpy as np
import torch

# %%
device = 'cuda'
bs = 2
folder = '..\\data\\Squid\\'


# %%
imgs_ = []
for i in range(1,6):
    img = nib.load(folder+'Squid_'+str(i)+'.nii')
    imgs_.append(img.get_fdata()[:,:,:,None])

V.multi_slice_viewer(imgs_[1])
# %%
for i in range(len(imgs_)):
    plt.subplot(1,len(imgs_),i+1)
    plt.imshow(imgs_[i].mean(2))
    plt.title('Squid '+ str(i+1))
# %%
imgs = [img[:,:,::10,None][:,:,:,[0,0,0,0]] for img in imgs_]
dataset = M.RhesusDataset(np.array(imgs[1:]).transpose([0,4,1,2,3]),device=device)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)


# %%

A = torch.tensor(imgs[0].transpose([3,0,1,2])).to(device).float()

model = M.PiecewiseRigid(sz=A.shape[1:],A=A,positions=None,centers=None,device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

losses = M.train_model(model,dataloader,optimizer,gamma=1e-3,epochs=100,device=device)

# %%
plt.plot(losses[30:])
plt.xlabel('Iterations')
plt.ylabel('Loss')
# %%
testloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
registered = np.zeros(list(A.shape[1:])+[4,len(dataset)])
for batch_idx, data in enumerate(testloader):
    x_t,flow,_,_ = model(data[0].to(device),data[1])
    x_t = x_t.detach().cpu().numpy()
    registered[:,:,:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(x_t,[2,3,4,1,0])

# %%
plt.subplot(1,2,1)
plt.imshow(np.array(imgs[1:]).mean(0)[:,:,:,0].mean(2))
plt.title('Before')
plt.subplot(1,2,2)
plt.imshow(registered.mean(4).mean(2)[:,:,0])
plt.title('After')

# %%
V.multi_slice_viewer(np.array(imgs[1:]).mean(0)[:,:,:,0].transpose(2,1,0))
V.multi_slice_viewer(registered.mean(4)[:,:,:,0].transpose(2,1,0))


