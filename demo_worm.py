# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 16:26:27 2022

@author: Amin
"""
from Methods.StatAtlas.Atlas.Atlas import Atlas
from torch.utils.data import DataLoader
from DataHandling import DataHandler
from Methods.ID import AutoDetect
from Methods import Preprocess
import torch.optim as optim
import visualization as V
import datasets as D
import models as M
import numpy as np
import torch
import glob

device = 'cuda'
# %%
save = False
file = '..\\results\\'
parallel = True

bs = 8
    
body = 'head'

# main_folder = 'C:\\Users\\Amin\\Desktop\\Projects\\Worm-Male-Atlas\\data\\Adult Male IDs\\'
# info_folder = 'C:\\Users\\Amin\\Desktop\\Projects\\Worm-Auto-ID\\codes\\Data\\NeuroPAL\\male_data.mat'
# subfolders = ['Heads\\','Tails\\']


main_folder = 'C:\\Users\\Amin\\Desktop\\Projects\\Worm-Auto-ID\\data\\Hobert\\Best otIs669 YA\\';
info_folder = 'C:\\Users\\Amin\\Desktop\\Projects\\Worm-Auto-ID\\codes\\Data\\NeuroPAL\\herm_data.mat'

subfolders = ['Best D-V\\Training Set 1\\',
              'Best D-V\\Training Set 2\\',
              'Best L-R\\Training Set 1\\',
              'Best L-R\\Training Set 2\\']

# %% Loading annotated images and training an atlas (male)



data = DataHandler.load_celegans_data(info_folder)
files = [glob.glob(main_folder+subfolder+'*_STRUCT.mat') for subfolder in subfolders]
files = [item for sublist in files for item in sublist]

ims = [DataHandler.load_neuropal_id(file) for file in files]
ims_body = [im for im in ims if im.bodypart == body]
files_body = [files[i] for i in range(len(files)) if ims[i].bodypart == body]

        
if body == 'head':
    neurons_body = list(set([neuron for ganglion in data['ganglia'][:9] for neuron in ganglion['neurons']]))
    neurons_body = neurons_body + ['AMSOL', 'AMSOR']

if body == 'tail':
    neurons_body = list(set([neuron for ganglion in data['ganglia'][13:21] for neuron in ganglion['neurons']]))
    neurons_body = neurons_body + ['PHSO1L', 'PHSO1R', 'AS10', 'DA7', 'VD11', 'VA11', 'PVPL', 'PVPR']


data_body = [Preprocess.zscore(DataHandler.load_neuropal(files_body[i][:-14]+'.mat')[0].astype(float)) for i in range(len(files_body))]

# %%
neurons,col,pos,_ = Atlas.sort_mu(ims_body,neurons_body,min_counts=len(ims_body))

scales = np.array([im.scale for im in ims_body])
pos = pos/scales.T[None,:,:]
col = col[:,:3,:]


# pos = pos / np.array([4,4,2])[None,:,None]
# data_body_mp = [d[::4,::4,::2,:3] for d in data_body]
# 

pos = pos / np.array([1,1,np.inf])[None,:,None]
data_body_mp = [d.max(2)[::1,::1,None,:3] for d in data_body]

max_shape = np.array([d.shape for d in data_body_mp]).max(0)

# %%
for f in range(len(data_body_mp)):
    pads = [(0,(max_shape[i]-data_body_mp[f].shape[i]).astype(int)) for i in range(4)]
    data_body_mp[f] = np.pad(data_body_mp[f],pads)

# %%
AutoDetect.visualize(np.array(data_body_mp).mean(0),
      np.hstack((pos[:,:,0],col[:,:,0])),
      factor=2,scale=1,microns=ims_body[f].scale[0])

# %%
# for i in range(len(data_body_mp)):
#     V.visualize_image(data_body_mp[i],
#           p=pos[:,:,i],c=col[:,:,i],
#           factor=1.5,scale=1,microns=ims_body[i].scale[0])


# %%
# n_rows,n_cols,n_secs = [1,2,1]
n_rows,n_cols,n_secs = [2,2,1]
tess = [np.array([(2*r+1)*max_shape[0]/(2*n_rows),(2*c+1)*max_shape[1]/(2*n_cols),(2*s+1)*max_shape[2]/(2*n_secs)]) 
        for r in range(n_rows) for c in range(n_cols) for s in range(n_secs)]
a = np.arange(n_rows*n_cols).reshape(n_rows,n_cols)
a[1::2,:] = a[1::2,::-1]
nbs = [[a_] for a_ in a.flatten().tolist()[1:]] + [[]]

# %%
dataset = D.PCImageDataset(np.array(data_body_mp[1:]).transpose([0,4,1,2,3]),pos,device=device)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

# %%
A = torch.tensor(data_body_mp[0].transpose([3,0,1,2])).to(device).float()

model = M.ImagePiecewiseRigid(sz=max_shape[:3],A=A,tess=tess,nbs=nbs,n_channels=3,
                          centers=torch.tensor(pos[:,:,0]),device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
M.train_model(model,dataloader,optimizer,gamma=1e3,epochs=100,epochs_theta=200,device=device)
    

# %%
testloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
registered = np.zeros(list(max_shape)+[len(dataset)])
pos_moved = np.zeros((pos.shape[0],3,len(dataset)))
for batch_idx, data in enumerate(testloader):
    x_t,flow,reg,moved = model(data[0].to(device),data[1])
    x_t = x_t.detach().cpu().numpy()
    registered[:,:,:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(x_t,[2,3,4,1,0])
    pos_moved[:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(torch.stack(moved).detach().cpu().numpy(),[1,2,0])
        
# %%

# AutoDetect.visualize(registered[:,:,:,:,1],
#       np.hstack((pos_moved[:,:,1],col[:,:,1])),scale=1,microns=ims_body[f].scale[0])

# AutoDetect.visualize(data_body_mp[2],
#       np.hstack((pos[:,:,2],col[:,:,1])),scale=1,microns=ims_body[f].scale[0])

# %%
# AutoDetect.visualize(registered.mean(4),
#       np.hstack((pos_moved.transpose([2,0,1]).reshape(pos_moved.shape[0]*pos_moved.shape[2],3),
#                   col[:,:,1:].transpose([2,0,1]).reshape(pos_moved.shape[0]*pos_moved.shape[2],4))),scale=1,microns=ims_body[f].scale[0],
#       save=save,file=file+'worm_'+body+'_img')

V.visualize_image(registered.mean(4),
      p=pos[:,:,0],c=col[:,:,0],names=neurons,
      factor=2,scale=1,microns=ims_body[0].scale[0])

V.visualize_image(np.array(data_body_mp).mean(0),
      p=pos[:,:,0],c=col[:,:,0],names=neurons,
      factor=2,scale=1,microns=ims_body[0].scale[0])


# AutoDetect.visualize(np.array(data_body_mp).mean(0),
#       np.hstack((pos.transpose([2,0,1]).reshape(pos.shape[0]*pos.shape[2],3),
#                  col.transpose([2,0,1]).reshape(pos.shape[0]*pos.shape[2],4))),scale=1,microns=ims_body[f].scale[0])

# AutoDetect.visualize(data_body_mp[0],
#       np.hstack((pos[:,:,0],col[:,:,0])),scale=1,microns=ims_body[f].scale[0])

