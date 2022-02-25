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
import datasets as D
import models as M
import numpy as np
import torch
import glob

device = 'cuda'
# %%
save = True
file = '..\\results\\'
parallel = True

bs = 12
    
body = 'head'

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
_,col,pos,_ = Atlas.sort_mu(ims_body,neurons_body)

scales = np.array([im.scale for im in ims_body])
pos = pos/scales.T[None,:,:]
# remove nans
col = col[~np.isnan(pos.sum(2).sum(1)),:,:]
pos = pos[~np.isnan(pos.sum(2).sum(1)),:,:]

pos = pos / np.array([4,4,np.inf])[None,:,None]
# pos[:,2,:] = 0

data_body_mp = [d.max(2)[::4,::4,None,:] for d in data_body]
max_shape = np.array([d.shape for d in data_body_mp]).max(0)

# %%
for f in range(len(data_body_mp)):
    pads = [(0,(max_shape[i]-data_body_mp[f].shape[i]).astype(int)) for i in range(4)]
    data_body_mp[f] = np.pad(data_body_mp[f],pads)

# %%
AutoDetect.visualize(np.array(data_body_mp).mean(0),
      np.hstack((pos[:,:,0],col[:,:,0])),scale=1,microns=ims_body[f].scale[0])
    
# %%
dataset = D.PCImageDataset(np.array(data_body_mp[1:]).transpose([0,4,1,2,3]),pos,device=device)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

# %%
A = torch.tensor(data_body_mp[0].transpose([3,0,1,2])).to(device).float()

model = M.PiecewiseRigid(sz=max_shape[:3],A=A,positions=torch.tensor(pos[:,:,1:]),
                      centers=torch.tensor(pos[:,:,0]),device=device)
# model = M.FlowField(sz=max_shape[:3],A=A,positions=torch.tensor(pos[:,:,1:]),
#                       centers=torch.tensor(pos[:,:,0]),device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
M.train_model(model,dataloader,optimizer,gamma=1e1,epochs=100,epochs_theta=200,device=device)
    

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

AutoDetect.visualize(registered.mean(4),
      np.hstack((pos_moved.transpose([2,0,1]).reshape(pos_moved.shape[0]*pos_moved.shape[2],3),
                 col[:,:,1:].transpose([2,0,1]).reshape(pos_moved.shape[0]*pos_moved.shape[2],4))),scale=1,microns=ims_body[f].scale[0],
      save=save,file=file+'worm_'+body+'_img')

AutoDetect.visualize(np.array(data_body_mp).mean(0),
      np.hstack((pos.transpose([2,0,1]).reshape(pos.shape[0]*pos.shape[2],3),
                 col.transpose([2,0,1]).reshape(pos.shape[0]*pos.shape[2],4))),scale=1,microns=ims_body[f].scale[0])

AutoDetect.visualize(data_body_mp[0],
      np.hstack((pos[:,:,0],col[:,:,0])),scale=1,microns=ims_body[f].scale[0])


# %%
dist = lambda x,y: np.sqrt(((x-y)**2).sum(1)).mean()


for i in range(len(pos)):
    print('After')
    print(np.cov(pos_moved[i,:2,:]))
    print('Before')
    print(np.cov(pos[i,:2,1:]))
    print('----')

# %%

# from Methods.Straightening import utils as U
# pos_moved_rigid = np.zeros((pos.shape[0],3,len(dataset)))
# registered_rigid = np.zeros(list(max_shape)+[len(dataset)])

# grid = torch.tensor(np.array(np.where(np.ones(list(max_shape[:3]))),dtype=float).T).float()
# pos_ = pos
# pos_[:,2,:] = 0
# for i in range(1,pos.shape[2]):
    
#     _,pos_moved_rigid[:,:,i-1],_ = U.procrustes(pos_[:,:,0], pos_[:,:,i], scaling=False)
#     _,_,tform = U.procrustes(pos_[:,:,i], pos_[:,:,0], scaling=False)
#     grid_ = M.PiecewiseRigid.extended_procrustes_flow(
#         [tform['rotation']],[tform['translation'][None,:]],torch.tensor([[0,0,0]]).float(),
#         grid,max_shape[:3].tolist(),std=10,normalize=False,device='cpu')
    
#     registered_rigid[:,:,:,:,i-1] = U.image_warp(data_body_mp[i], grid_.numpy())

# # %%
# for i in range(len(pos)):
#     print('After')
#     print(np.cov(pos_moved[i,:2,:]))
#     print('Before')
#     print(np.cov(pos[i,:2,1:]))
#     print('Analytical')
#     print(np.cov(pos_moved_rigid[i,:2,:]))
#     print('----')


# AutoDetect.visualize(registered.mean(4),
#       np.hstack((pos_moved.transpose([2,0,1]).reshape(66*12,3),
#                  col[:,:,1:].transpose([2,0,1]).reshape(66*12,4))),scale=1,microns=ims_body[f].scale[0])

# AutoDetect.visualize(registered_rigid.mean(4),
#       np.hstack((pos_moved_rigid.transpose([2,0,1]).reshape(66*12,3),
#                  col[:,:,1:].transpose([2,0,1]).reshape(66*12,4))),scale=1,microns=ims_body[f].scale[0])

