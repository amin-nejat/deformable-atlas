# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 11:36:41 2022

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

# %% Joint atlas
side = 'L'
imgs = []
for sex in ['F','M']:
    for genotype in ['egfr','samw','star','tkv']:
        file_name = genotype+'_'+sex+'_'+side

        files = glob.glob(folder+file_name+'_lei_2X*')[:16]
        imgs += [1-np.array(Image.open(file))[100:-100:4,100:-100:4,None,None,1].astype(float)/255 for file in files]

sz = np.array(imgs[0].shape[:3])

# %%
def train_atlas(train,test,tess,nbs,bs=12,gamma=1e-6,epochs=150,epochs_theta=200):

    train_dataset = D.ImageDataset(np.array(train[1:]).transpose([0,4,1,2,3]),device=device)
    dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=False, num_workers=0)
    
    A = torch.tensor(train[0].transpose([3,0,1,2])).to(device).float()
    
    model = M.ImagePiecewiseRigid(sz=sz,A=A,reg_pc=False,tess=tess,nbs=nbs,n_channels=1,device=device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_loss = M.train_model(model,dataloader,optimizer,gamma=gamma,epochs=epochs,epochs_theta=epochs_theta,device=device)
    
    test_dataset = D.ImageDataset(np.array(test).transpose([0,4,1,2,3]),device=device)
    testloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)
    registered = np.zeros(list(A.shape[1:])+[1,len(test_dataset)])
    
    test_loss = []
    for batch_idx, data in enumerate(testloader):
        with torch.no_grad():
            x_t,flow,_,_ = model(data[0].to(device),data[1])
            test_loss.append(model.observation_loss(x_t).item())
            x_t = x_t.detach().cpu().numpy()
            registered[:,:,:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(x_t,[2,3,4,1,0])
        
    return model, registered, train_loss, np.array(test_loss)


# %%
n_rows,n_cols,n_secs = [3,3,1]
tess = [np.array([(2*r+1)*sz[0]/(2*n_rows),(2*c+1)*sz[1]/(2*n_cols),(2*s+1)*sz[2]/(2*n_secs)]) 
        for r in range(n_rows) for c in range(n_cols) for s in range(n_secs)]
a = np.arange(n_rows*n_cols).reshape(n_rows,n_cols)
a[1::2,:] = a[1::2,::-1]
nbs = [[a_] for a_ in a.flatten().tolist()[1:]] + [[]]

# %%
train_index,test_index = np.arange(128), np.arange(128)

model, registered, trl, tel = train_atlas(
            [imgs[idx] for idx in train_index],
            [imgs[idx] for idx in test_index],
            tess,nbs,bs=12,gamma=1e-6,epochs=40,epochs_theta=200)

V.visualize_image(registered.mean(4),scale=1,
      save=save,file=file+'joint_atlas',titlestr='Atlas')

# %%
jacs_sum,jacs,titles = [],[],[]

for sex in ['F','M']:
    for genotype in ['egfr','samw','star','tkv']:
        file_name = genotype+'_'+sex+'_'+side

        files = glob.glob(folder+file_name+'_lei_2X*')[:16]
        imgs_ = [1-np.array(Image.open(file))[100:-100:4,100:-100:4,None,None,1].astype(float)/255 for file in files]

        test_dataset = D.ImageDataset(np.array(imgs_).transpose([0,4,1,2,3]),device=device)
        testloader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0)
        
        jac,jac_sum = [],[]
        for batch_idx, data in enumerate(testloader):
            with torch.no_grad():
                jac_ = model.det_jac(data[0].to(device),data[1]).abs().log()
                jac += jac_.detach().cpu().numpy().tolist()
                # jac = jac_.sum(0) if len(jac) == 0 else jac + jac_.sum(0)
                # jac += [jac_[i].min().item() for i in range(jac_.shape[0])]
                # jac += torch.quantile(jac_.view(jac_.shape[0],-1),.1,dim=1).detach().cpu().numpy().tolist()
                jac_sum += jac_.mean([1,2,3]).detach().cpu().numpy().tolist()
        jacs += [jac]
        # jacs += [jac.detach().cpu().numpy()/len(test_dataset)]
        jacs_sum += [np.array(jac_sum)]
        titles += [file_name]
        
# %%
V.plot_bar(jacs_sum,ticks=titles,titlestr='Average Det. of Jac.',save=save,file=file+'fly_det_jac')
V.visualize_subjects(jacs, titles)

# %%
a = np.array(jacs[0]+jacs[1]+jacs[2]+jacs[3])
b = np.array(jacs[4]+jacs[5]+jacs[6]+jacs[7])
t_stat = (a.mean(0) - b.mean(0))/np.sqrt(a.var(0) + b.var(0))

V.visualize_subjects([t_stat], ['M vs. F'])