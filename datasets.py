# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:53:52 2022

@author: Amin
"""
from torch.utils.data import Dataset
import torch

# %%
class PCDataset(Dataset):
    def __init__(self,positions,device='cuda'):
        self.positions = torch.tensor(positions).to(device).float()
        
    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.positions[idx,:,:]

        return sample,idx
    
# %%
class ImageDataset(Dataset):
    def __init__(self,imgs,device='cuda'):
        self.imgs = torch.tensor(imgs).to(device).float()
        
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.imgs[idx,:,:,:,:]
        sample[sample<0] = 0

        return sample,idx
    
# %%
class PCImageDataset(Dataset):
    def __init__(self,imgs,positions,device='cuda'):
        self.imgs = torch.tensor(imgs).to(device).float()
        self.positions = torch.tensor(positions).to(device).float()
        
    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.imgs[idx,:,:,:,:]
        sample[sample<0] = 0

        return sample,self.positions[:,:,idx]