# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 19:00:49 2022

@author: Amin
"""

from Methods.StatAtlas.Atlas.Atlas import Atlas
from torch.utils.data import DataLoader
from DataHandling import DataHandler
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

bs = 12
    
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

# if body == 'head':
#     indices = [0,1,2,3,4,5,6,12,15,16,17,18]
#     ims_body = [ims_body[i] for i in indices]
#     files_body = [files_body[i] for i in indices]

        
if body == 'head':
    neurons_body = list(set([neuron for ganglion in data['ganglia'][:9] for neuron in ganglion['neurons']]))
    neurons_body = neurons_body + ['AMSOL', 'AMSOR']

if body == 'tail':
    neurons_body = list(set([neuron for ganglion in data['ganglia'][13:21] for neuron in ganglion['neurons']]))
    neurons_body = neurons_body + ['PHSO1L', 'PHSO1R', 'AS10', 'DA7', 'VD11', 'VA11', 'PVPL', 'PVPR']

# %%
neurons,col,pos,_ = Atlas.sort_mu(ims_body,neurons_body,min_counts=len(ims_body))

scales = np.array([im.scale for im in ims_body])
pos = pos*scales.T[None,:,:]
# remove nans
# neurons_body = [neurons_body[i] for i in np.where(~np.isnan(pos.sum(2).sum(1)))[0]]
# col = col[~np.isnan(pos.sum(2).sum(1)),:,:]
# pos = pos[~np.isnan(pos.sum(2).sum(1)),:,:]

col[col < 0] = 0
# col = col/np.nanmax(col)
point_cloud = np.hstack((pos,col[:,:3])).transpose([2,1,0])
V.visualize_pc(point_cloud)
# %%
dataset = D.PCDataset(point_cloud[1:,:,:],device=device)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

# %%
sz = torch.tensor([pos[:,i,:].max() for i in range(3)])
model = M.PCPiecewiseRigid(A=point_cloud[9],sz=sz,device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
M.train_model(model,dataloader,optimizer,gamma=1e0,epochs=10,device=device)

# %%
testloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
pos_moved = np.zeros((len(dataset),6,pos.shape[0]))
for batch_idx, data in enumerate(testloader):
    x_t,_,_,_ = model(data[0].to(device),data[1])
    pos_moved[batch_idx*bs:batch_idx*bs+x_t.shape[0],:,:] = x_t.detach().cpu().numpy()
    
# %%
V.visualize_pc(pos_moved)
# %%
dist = lambda x,y: np.sqrt(((x-y)**2).sum(1)).mean()

for i in range(len(pos)):
    print('After')
    print(np.cov(pos_moved[:,:3,i].T))
    print('Before')
    print(np.cov(pos[i,:3,1:]))
    print('----')
    
# %%
# Atlas.visualize_pretty({'mu':torch.cat((model.theta_l,model.theta_c),1).detach().cpu().numpy(),
#                         'sigma':np.array([np.cov(pos_moved[:,:,i].T) for i in range(len(pos))]).transpose(1,2,0),
#                         'names':neurons}, 
#                        pos_moved.transpose(0,2,1), '')

# %%
# pos_sample = torch.cat([torch.cat((model.prior_l.sample(),model.prior_c.sample()),1).T[None,:,:] for i in range(10)],0).cpu().numpy()

# Atlas.visualize_pretty({'mu':pos_sample.mean(0).T,
#                         'sigma':np.array([np.cov(pos_sample[:,:,i].T) for i in range(len(pos))]).transpose(1,2,0),
#                         'names':neurons}, 
#                        pos_sample.transpose(0,2,1), '')

# %%
Atlas.visualize_pretty({'mu':pos_moved.mean(0).T,
                        'sigma':np.array([np.cov(pos_moved[:,:,i].T)/3 for i in range(len(pos))]).transpose(1,2,0),
                        'names':neurons}, 
                       pos_moved.transpose(0,2,1), '',save=save,file=file+'worm_'+body+'_pc')

# %%
Atlas.visualize_pretty({'mu':point_cloud.mean(0).T,
                        'sigma':np.array([np.cov(point_cloud[:,:,i].T) for i in range(len(pos))]).transpose(1,2,0),
                        'names':neurons}, 
                          point_cloud.transpose(0,2,1), '')