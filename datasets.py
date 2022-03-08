# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:53:52 2022

@author: Amin
"""
from torch.utils.data import Dataset
import torch

# %%
class PCDataset(Dataset):
    '''Point cloud dataset, to store and read point cloud data in batches.
    '''
    
    def __init__(self,positions,device='cuda'):
        '''Constructor
        
        Parameters
        ----------
        positions (np.ndarray): B.6.N where B is the dataset size and N is the number
            of points in the point cloud. 6 corresponds to (XYZ RGB).
        device (string): Either 'cpu' or 'cuda' (default).
        '''
        
        self.positions = torch.tensor(positions).to(device).float()
        
    def __len__(self):
        '''Read out the size of the dataset.
        '''
        return self.positions.shape[0]

    def __getitem__(self,idx):
        '''Read a batch indexed by idx.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.positions[idx,:,:]

        return sample,idx
    
# %%
class ImageDataset(Dataset):
    '''Image dataset, to store and read image data in batches.
    '''
    def __init__(self,imgs,device='cuda'):
        '''Constructor
        
        Parameters
        ----------
        imgs (np.ndarray): B.X.Y.Z.C where B is the dataset size, X, Y, Z are 
            image dimensions along three axes and C is the number of color channels.
        device (string): Either 'cpu' or 'cuda' (default).
        '''
        self.imgs = torch.tensor(imgs).to(device).float()
        
    def __len__(self):
        '''Read out the size of the dataset.
        '''
        return self.imgs.shape[0]

    def __getitem__(self,idx):
        '''Read a batch indexed by idx.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.imgs[idx,:,:,:,:]
        sample[sample<0] = 0

        return sample,idx
    
# %%
class PCImageDataset(Dataset):
    '''Image and point cloud dataset, to store and read joint datasets of images
        and point clouds.
    '''
    def __init__(self,imgs,positions,device='cuda'):
        '''Constructor
        
        Parameters
        ----------
        imgs (np.ndarray): B.X.Y.Z.C where B is the dataset size, X, Y, Z are 
            image dimensions along three axes and C is the number of color channels.
        positions (np.ndarray): B.6.N where B is the dataset size and N is the number
            of points in the point cloud. 6 corresponds to (XYZ RGB).
        device (string): Either 'cpu' or 'cuda' (default).
        '''
        self.imgs = torch.tensor(imgs).float().to(device)
        self.positions = torch.tensor(positions).to(device).float()
        
    def __len__(self):
        '''Read out the size of the dataset.
        '''
        return self.imgs.shape[0]

    def __getitem__(self,idx):
        '''Read a batch of images and corresponding point clouds indexed by idx.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.imgs[idx,:,:,:,:]
        sample[sample<0] = 0

        return sample,self.positions[:,:,idx]