# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 11:53:58 2022

@author: Amin
"""

from Methods.StatAtlas.Atlas.Atlas import Atlas
from sklearn.model_selection import KFold
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

device = 'cuda'
# %%

save = True
file = '..\\results\\'

bs = 12
body = 'tail'


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
V.visualize_pc(point_cloud)

# %%
tess = [[neurons.index(n) for n in ganglion if n in neurons] for ganglion in ganglia_body]

if body == 'head': tess = [tess[0]+tess[1]+tess[2]+tess[3]+tess[4],tess[4]+tess[5]+tess[6]+tess[7],tess[7]+tess[8]]
if body == 'tail': tess = [tess[4]+tess[0],tess[0]+tess[1],tess[1]+tess[2]+tess[3]+tess[5]]

mask = 1-np.eye(len(tess))

# %%
def train_atlas(train,test,tess,mask,bs=12,gamma=1e3,epochs=100,epochs_theta=50,color_model='dirichlet'):
    train_dataset = D.PCDataset(train,device=device)
    dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=0)

    sz = torch.tensor([pos[:,i,:].max() for i in range(3)])
    if color_model == 'normal':
        model = M.PCPiecewiseRigidNormal(A=point_cloud[0],sz=sz,tess=tess,mask=mask,device=device)
    else:
        model = M.PCPiecewiseRigid(A=point_cloud[0],sz=sz,tess=tess,mask=mask,device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train_loss = M.train_model(model,dataloader,optimizer,gamma=gamma,
               epochs=epochs,epochs_theta=epochs_theta,device=device)
    
    test_dataset = D.PCDataset(test,device=device)
    testloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)
    pos_moved = np.zeros((len(test_dataset),6,pos.shape[0]))
    test_loss = []
    for batch_idx, data in enumerate(testloader):
        with torch.no_grad():
            x_t,_,_,_ = model(data[0].to(device),data[1])
            pos_moved[batch_idx*bs:batch_idx*bs+x_t.shape[0],:,:] = x_t.detach().cpu().numpy()
            test_loss.append(model.observation_loss(x_t).item())
            
    return model, pos_moved, train_loss, np.array(test_loss)

# %% prigid dirichlet
kf = KFold(n_splits=5)

models_pr, aligned_pr, trl_pr, tel_pr = [], [], [], []
for i, indices in enumerate(kf.split(np.arange(point_cloud.shape[0]))):
    train_index,test_index = indices
    model_, aligned_, trl_, tel_ = train_atlas(point_cloud[train_index,:,:],
                point_cloud[test_index,:,:],tess,mask,bs=12,gamma=1e-5,epochs=100,epochs_theta=10)
    models_pr.append(model_); aligned_pr.append(aligned_); trl_pr.append(trl_), tel_pr.append(tel_)
    
# %% rigid dirichlet
kf = KFold(n_splits=5)

models_r, aligned_r, trl_r, tel_r = [], [], [], []
for i, indices in enumerate(kf.split(np.arange(point_cloud.shape[0]))):
    train_index,test_index = indices
    model_, aligned_, trl_, tel_ = train_atlas(point_cloud[train_index,:,:],
                point_cloud[test_index,:,:],None,None,bs=12,gamma=0,epochs=100,epochs_theta=10)
    models_r.append(model_); aligned_r.append(aligned_); trl_r.append(trl_), tel_r.append(tel_)


# %% prigid normal
kf = KFold(n_splits=5)

models_pr_n, aligned_pr_n, trl_pr_n, tel_pr_n = [], [], [], []
for i, indices in enumerate(kf.split(np.arange(point_cloud.shape[0]))):
    train_index,test_index = indices
    model_, aligned_, trl_, tel_ = train_atlas(point_cloud[train_index,:,:],
                point_cloud[test_index,:,:],tess,mask,bs=12,gamma=1e-5,epochs=100,epochs_theta=10,
                color_model='normal')
    models_pr_n.append(model_); aligned_pr_n.append(aligned_); trl_pr_n.append(trl_), tel_pr_n.append(tel_)


# %% rigid normal
kf = KFold(n_splits=5)

models_r_n, aligned_r_n, trl_r_n, tel_r_n= [], [], [], []
for i, indices in enumerate(kf.split(np.arange(point_cloud.shape[0]))):
    train_index,test_index = indices
    model_, aligned_, trl_, tel_ = train_atlas(point_cloud[train_index,:,:],
                point_cloud[test_index,:,:],None,None,bs=12,gamma=0,epochs=100,epochs_theta=10,
                color_model='normal')
    models_r_n.append(model_); aligned_r_n.append(aligned_); trl_r_n.append(trl_), tel_r_n.append(tel_)

# %%
V.plot_loss([np.array(trl_r).T,np.array(trl_r_n).T,np.array(trl_pr).T,np.array(trl_pr_n).T],
            labels=['PR-Dir','R-Dir','PR-Nor','R-Nor'],titlestr='Training Loss',
            save=True,file=file+'xval_train_dir')

# %%
V.plot_bar([np.array(tel_pr).squeeze(),np.array(tel_r).squeeze(),np.array(tel_pr_n).squeeze(),np.array(tel_r_n).squeeze()]
           ,ticks=['PR-Dir','R-Dir','PR-Nor','R-Nor'],
           titlestr='Test Likelihood of Models (Cross Validated)',
           ylabel='NLL',save=True,file=file+'nll_models')