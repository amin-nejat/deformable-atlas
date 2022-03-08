# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:41:57 2022

@author: Amin
"""
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
import pyro.distributions as dist
from scipy.io import loadmat
import torch.optim as optim
import visualization as V
import datasets as D
import models as M
import numpy as np
import scipy as sp
import torch
import pyro

# %%

content = loadmat('data//atlas_xx_rgb.mat',simplify_cells=True)
neurons = list(content['atlas']['tail']['N'])
mu = content['atlas']['tail']['model']['mu']
sigma = content['atlas']['tail']['model']['sigma']

data = loadmat('data//herm_data.mat',simplify_cells=True)
neurons_body = list(set([neuron for ganglion in data['ganglia'][13:21] for neuron in ganglion['neurons']]))
neurons_body = neurons_body + ['PHSO1L', 'PHSO1R', 'AS10', 'DA7', 'VD11', 'VA11', 'PVPL', 'PVPR']

ganglia_body = [list(set([neuron for neuron in ganglion['neurons']])) for ganglion in data['ganglia'][13:]]
ganglia_body += [['AS10', 'DA7', 'VD11','VA11']]
ganglia_body += [['PHSO1R','PHSO1L']]
    
tess = [[neurons.index(n) for n in ganglion if n in neurons] for ganglion in ganglia_body]
tess = [tess[4]+tess[0],tess[0]+tess[1],tess[1]+tess[2]+tess[3]+tess[5]]

mask = 1-np.eye(len(tess))

# %%
def sample(theta_l,theta_c,sigma_l,n_sample=10):
    with pyro.plate('neurons',theta_c.size(0)):
        with pyro.plate('data',n_sample):
            sample_c = pyro.sample('obs_c',dist.Dirichlet(theta_c))
            sample_l = pyro.sample('obs_l',dist.MultivariateNormal(theta_l,sigma_l*torch.eye(3)))
    return torch.cat((sample_l,sample_c),2).permute(0,2,1)
        
# %%
theta_l=torch.tensor(mu[:,:3]).float()
theta_c=torch.tensor(sp.special.softmax(mu[:,3:]/50,axis=1)).float()
sigma_l=torch.ones((1)).float()
point_cloud = sample(theta_l,theta_c,sigma_l)
for i in range(point_cloud.shape[0]):
    rot = torch.tensor(R.from_rotvec(np.array([0,0,np.random.randn()/3])).as_matrix()).float()
    point_cloud[i,:3,:] = rot@point_cloud[i,:3,:]

# %%
device = 'cuda'
bs = 16

dataset = D.PCDataset(point_cloud[1:,:,:],device=device)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)

# %% train
sz = torch.tensor([point_cloud[:,i,:].max() for i in range(3)])
model = M.PCPiecewiseRigid(A=point_cloud[0],sz=sz,tess=tess,mask=mask,device=device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
M.train_model(model,dataloader,optimizer,gamma=1e0,epochs=100,epochs_theta=50,device=device)

# %% test
testloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=0)
pos_moved = np.zeros((len(dataset),6,point_cloud.shape[2]))
for batch_idx, data in enumerate(testloader):
    x_t,_,_,_ = model(data[0].to(device),data[1])
    pos_moved[batch_idx*bs:batch_idx*bs+x_t.shape[0],:,:] = x_t.detach().cpu().numpy()
    
# %% vis
V.visualize_pc(pos_moved,atlas=model.A.T.detach().cpu().numpy(),title_str='Inferred Atlas')
V.visualize_pc(point_cloud,title_str='Pre Registration')

