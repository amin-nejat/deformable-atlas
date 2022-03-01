# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 10:03:26 2022

@author: Amin
"""
from Methods.StatAtlas.Atlas.Atlas import Atlas
from torch.utils.data import DataLoader
from DataHandling import DataHandler
from Methods import Preprocess
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
save = True
file = '..\\results\\'
bs = 12
body = 'tail'
transformation = 'PR'

main_folder = 'C:\\Users\\Amin\\Desktop\\Projects\\Worm-Male-Atlas\\data\\Adult Male IDs\\'
info_folder = 'C:\\Users\\Amin\\Desktop\\Projects\\Worm-Auto-ID\\codes\\Data\\NeuroPAL\\male_data.mat'

# %% Loading annotated images and training an atlas (male)
subfolders = ['Heads\\','Tails\\']

data = DataHandler.load_celegans_data(info_folder)
files = [glob.glob(main_folder+subfolder+'*_STRUCT.mat') for subfolder in subfolders]
files = [item for sublist in files for item in sublist]

ims = [DataHandler.load_neuropal_id(file) for file in files]
ims_body = [im for im in ims if im.bodypart == body]
files_body = [files[i] for i in range(len(files)) if ims[i].bodypart == body]

if body == 'head':
    indices = [0,1,2,3,4,5,6,12,15,16,17,18]
    ims_body = [ims_body[i] for i in indices]
    files_body = [files_body[i] for i in indices]
        
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
col = sp.special.softmax(col[:,:3,:],axis=1)


pos = pos / np.array([4,4,np.inf])[None,:,None]

data_body_mp = [d.max(2)[::4,::4,None,:3] for d in data_body]
max_shape = np.array([d.shape for d in data_body_mp]).max(0)

for f in range(len(data_body_mp)):
    pads = [(0,(max_shape[i]-data_body_mp[f].shape[i]).astype(int)) for i in range(4)]
    data_body_mp[f] = np.pad(data_body_mp[f],pads)

# %%
def train_atlas(train,test,pos_train,pos_test,tess,nbs,bs=12,gamma=1e1,epochs=100,epochs_theta=50):

    train_dataset = D.PCImageDataset(np.array(train[1:]).transpose([0,4,1,2,3]),pos_train[:,:,1:],device=device)
    dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=0)
    
    A = torch.tensor(train[0].transpose([3,0,1,2])).to(device).float()
    
    model = M.ImagePiecewiseRigid(sz=max_shape[:3],A=A,tess=tess,nbs=nbs,n_channels=3,
                                  centers=torch.tensor(pos_train[:,:,0]),device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loss = M.train_model(model,dataloader,optimizer,gamma=gamma,epochs=epochs,epochs_theta=epochs_theta,device=device)
    
    test_dataset = D.PCImageDataset(np.array(test).transpose([0,4,1,2,3]),pos_test,device=device)
    testloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)
    registered = np.zeros(list(max_shape)+[len(test_dataset)])
    pos_moved = np.zeros((pos_test.shape[0],3,len(test_dataset)))
    
    test_loss = []
    
    for batch_idx, data in enumerate(testloader):
        with torch.no_grad():
            x_t,flow,reg,moved = model(data[0].to(device),data[1])
            test_loss.append(model.observation_loss(x_t).item())
            x_t = x_t.detach().cpu().numpy()
            registered[:,:,:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(x_t,[2,3,4,1,0])
            pos_moved[:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(torch.stack(moved).detach().cpu().numpy(),[1,2,0])
            
    return model, registered, pos_moved, train_loss, np.array(test_loss)
# %%
if transformation == 'PR':
    n_rows,n_cols,n_secs = [2,2,1]
    tess = [np.array([(2*r+1)*max_shape[0]/(2*n_rows),(2*c+1)*max_shape[1]/(2*n_cols),(2*s+1)*max_shape[2]/(2*n_secs)]) 
            for r in range(n_rows) for c in range(n_cols) for s in range(n_secs)]
    nbs = [[1],[3],[2],[]]

if transformation == 'R':
    tess,nbs = None,None

models, aligned, trl, tel = [], [], [], []

# train_index,test_index = np.arange(len(data_body_mp)-5), np.arange(len(data_body_mp)-5,len(data_body_mp))
train_index,test_index = np.arange(len(data_body_mp)), np.arange(len(data_body_mp))

errors = []
n_neurons = [5,10,20]
for i in range(len(n_neurons)):
    model_, registered_, aligned_, trl_, tel_ = train_atlas(
                [data_body_mp[idx] for idx in train_index],
                [data_body_mp[idx] for idx in train_index],
                pos[:n_neurons[i],:,train_index],
                pos[:,:,train_index],
                tess,nbs,bs=12,gamma=1e1,epochs=100,epochs_theta=200)
    models.append(model_); aligned.append(aligned_); trl.append(trl_), tel.append(tel_)
    
    errors.append(np.sqrt(((aligned_[20:,:,0][:,:,None]-aligned_[20:,:,test_index[1:]])**2).sum(1)).mean(0))

    V.visualize_image(registered_.mean(4),scale=1,microns=ims_body[f].scale[0],
                      p=pos[:n_neurons[i],:,train_index[0]],
                      c=col[:n_neurons[i],:3,train_index[0]],
                      names=neurons[:n_neurons[i]],olp=True,tol=1e-10,factor=2,
          save=save,file=file+'worm_'+body+'_img'+'_'+str(n_neurons[i])+'_neurons_'+transformation,
          titlestr=str(n_neurons[i])+' Neurons')

V.visualize_image(np.array([data_body_mp[idx] for idx in test_index]).mean(0),
                  p=pos[:n_neurons[i],:,train_index[0]],
                  c=col[:n_neurons[i],:3,train_index[0]],
                  names=neurons[:n_neurons[i]],olp=True,tol=1e-10,factor=2,
                  scale=1,microns=ims_body[f].scale[0],save=save,
                  file=file+'worm_'+body+'_img'+'_unregistered_'+transformation,
                  titlestr='Unregistered')
# %%
V.plot_loss(trl,
            labels=['5','10','20'],titlestr='Training Loss',
            save=save,file=file+'image_train_dir_'+transformation)

# %%
V.plot_bar(errors,ticks=['5','10','20'],
           titlestr='Error with More Annotations',
           ylabel='Error',save=save,file=file+'semi_supervised_error_'+transformation)