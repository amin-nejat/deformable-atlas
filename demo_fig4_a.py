# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 23:38:07 2022

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
sex = 'F'
genotype = 'egfr'
side = 'L'

file_name = genotype+'_'+sex+'_'+side

files = glob.glob(folder+file_name+'_lei_2X*')[:64]
imgs = [1-np.array(Image.open(file))[100:-100:4,100:-100:4,None,None,1].astype(float)/255 for file in files]


idx_show = [0,1,7]
V.visualize_subjects([imgs[i] for i in idx_show],titles=[file_name+str(i) for i in range(len(idx_show))],
                     save=save,file=file+file_name+'_subjects')

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
n_rows,n_cols,n_secs = [2,2,1]
tess = [np.array([(2*r+1)*sz[0]/(2*n_rows),(2*c+1)*sz[1]/(2*n_cols),(2*s+1)*sz[2]/(2*n_secs)]) 
        for r in range(n_rows) for c in range(n_cols) for s in range(n_secs)]
a = np.arange(n_rows*n_cols).reshape(n_rows,n_cols)
a[1::2,:] = a[1::2,::-1]
nbs = [[a_] for a_ in a.flatten().tolist()[1:]] + [[]]

# %%
train_index,test_index = np.arange(32), np.arange(32)

model, registered, trl, tel = train_atlas(
            [imgs[idx] for idx in train_index],
            [imgs[idx] for idx in test_index],
            tess,nbs,bs=12,gamma=1e-6,epochs=100,epochs_theta=200)

V.visualize_image(registered.mean(4),scale=1,
      save=save,file=file+file_name,titlestr='Atlas')

V.visualize_image(np.array([imgs[idx] for idx in test_index]).mean(0),
                  scale=1,save=save,file=file+file_name+'_unregistered',
                  titlestr='Unregistered')

# %%
V.plot_loss([np.array(trl).T],labels=['PR'],titlestr='Training Loss',
            save=save,file=file+file_name+'_train_loss')

