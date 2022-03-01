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
import scipy as sp
import torch
import glob

# %%
device = 'cuda'
save = False
load_model = False
body = 'head'
transformation = 'R'
bs = 12
file = '..\\results\\'

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
    ganglia_body = [list(set([neuron for neuron in ganglion['neurons']])) for ganglion in data['ganglia'][:9]]
    ganglia_body += [['AMSOR','AMSOL']]

if body == 'tail':
    neurons_body = list(set([neuron for ganglion in data['ganglia'][13:21] for neuron in ganglion['neurons']]))
    neurons_body = neurons_body + ['PHSO1L', 'PHSO1R', 'AS10', 'DA7', 'VD11', 'VA11', 'PVPL', 'PVPR']
    
    ganglia_body = [list(set([neuron for neuron in ganglion['neurons']])) for ganglion in data['ganglia'][13:]]
    ganglia_body += [['AS10', 'DA7', 'VD11','VA11']]
    ganglia_body += [['PHSO1R','PHSO1L']]

# %%
neurons,col,pos,_ = Atlas.sort_mu(ims_body,neurons_body,min_counts=len(ims_body))

scales = np.array([im.scale for im in ims_body])
pos = pos*scales.T[None,:,:]
point_cloud = np.hstack((pos,sp.special.softmax(col[:,:3,:],axis=1))).transpose([2,1,0])

# %% ganglion tesselations
tess = [[neurons.index(n) for n in ganglion if n in neurons] for ganglion in ganglia_body]

if body == 'head':
    tess = [tess[0]+tess[1]+tess[2]+tess[3]+tess[4],tess[4]+tess[5]+tess[6]+tess[7],tess[7]+tess[8]]
    
if body == 'tail':
    tess = [tess[4]+tess[0],tess[0]+tess[1],tess[1]+tess[2]+tess[3]+tess[5]]

mask = 1-np.eye(len(tess))

if transformation == 'R':
    tess,mask = None,None

# %% create dataset
dataset = D.PCDataset(point_cloud[1:,:,:],device=device)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

# %% train
sz = torch.tensor([pos[:,i,:].max() for i in range(3)])
model = M.PCPiecewiseRigid(A=point_cloud[0],sz=sz,tess=tess,mask=mask,device=device)

# Load saved model
if load_model:
    model.load_state_dict(torch.load(file+'worm_'+body+'_'+transformation))
    model.eval()
else:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    M.train_model(model,dataloader,optimizer,gamma=1e-6,epochs=100,epochs_theta=10,device=device)
    if save:
        torch.save(model.state_dict(), file+'worm_'+body+'_'+transformation) # Save

# %% test
testloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
pos_moved = np.zeros((len(dataset),6,pos.shape[0]))
for batch_idx, data in enumerate(testloader):
    x_t,_,_,_ = model(data[0].to(device),data[1])
    pos_moved[batch_idx*bs:batch_idx*bs+x_t.shape[0],:,:] = x_t.detach().cpu().numpy()
    
# %% vis
V.visualize_pc([],atlas=model.A.T.detach().cpu().numpy(),olp=False,
               save=save,file=file+'worm_'+body+'_3d_'+transformation)
    
# %%
Atlas.visualize_pretty({'mu':pos_moved.mean(0).T,
                        'sigma':np.array([np.cov(pos_moved[:,:,i].T)/3 for i in range(len(pos))]).transpose(1,2,0),
                        'names':neurons}, 
                       pos_moved.transpose(0,2,1), '',
                       olp=True,tol=8e-1,fontsize=12,
                       save=save,file=file+'worm_'+body+'_pc_'+transformation)
