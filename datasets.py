# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:53:52 2022

@author: Amin
"""
from turtle import color
from torch.utils.data import Dataset

import torch.nn as nn
import torch

from scipy.io import loadmat

import numpy as np

import utils
# import h5py
# import os


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


class NeuroPALDataset(Dataset):
    @staticmethod
    def zscore(data):
        """Z-scoring the image independently on each color channel"""
        zscored = data.copy()*0
        for d in range(data.shape[3]):
            zscored[:,:,:,d] = (data[:,:,:,d] - data[:,:,:,d].mean())/data[:,:,:,d].std()
            
        return zscored
        
    @staticmethod
    def load_neuropal_id(file):
        """Load NeuroPAL meta data"""
        content = loadmat(file,simplify_cells=True)['image']
        im = Image(bodypart=content['bodypart'].lower(),scale=content['scale'])
        im.neurons = []
        for n in range(len(content['neurons'])):
            neuron = Neuron()
            
            neuron.position                = content['neurons'][n]['position']-1 # Matlab to Python
            neuron.color                   = content['neurons'][n]['color']
            neuron.color_readout           = content['neurons'][n]['color_readout']
            neuron.covariance              = content['neurons'][n]['covariance']
            neuron.annotation              = content['neurons'][n]['annotation'] if len(content['neurons'][n]['annotation']) > 0 else ''
            neuron.annotation_confidence   = content['neurons'][n]['annotation_confidence']
            neuron.baseline                = content['neurons'][n]['baseline']
            neuron.deterministic_id        = content['neurons'][n]['deterministic_id']
            neuron.probabilistic_ids       = content['neurons'][n]['probabilistic_ids']
            neuron.probabilistic_probs     = content['neurons'][n]['probabilistic_probs']
            neuron.rank                    = content['neurons'][n]['rank']

            im.neurons.append(neuron)
        
        return im

    @staticmethod
    def load_neuropal(file):
        """Load a NeuroPAL formatted file"""
        content = loadmat(file,simplify_cells=True)

        scale = content['info']['scale'].T
        rgbw = content['info']['RGBW']
        
        image = content['data'][:,:,:,rgbw-1].squeeze()
        body = content['worm']['body'].lower()
        return image, scale, body

    @staticmethod
    def load_celegans_data(file):
        content = loadmat(file,simplify_cells=True)
        return content

    def __init__(self,info,files,body,df=[1,1,1],margin=[10,10,3]):
        data = NeuroPALDataset.load_celegans_data(info)
        ims = [NeuroPALDataset.load_neuropal_id(file) for file in files]

        ims_body = [im for im in ims if im.bodypart == body]
        files_body = [files[i] for i in range(len(files)) if ims[i].bodypart == body]
        scales = np.array([im.scale for im in ims_body])


        if body == 'head':
            neurons_body = list(set([neuron for ganglion in data['ganglia'][:9] for neuron in ganglion['neurons']]))
            neurons_body = neurons_body + ['AMSOL', 'AMSOR']

        if body == 'tail':
            neurons_body = list(set([neuron for ganglion in data['ganglia'][13:21] for neuron in ganglion['neurons']]))
            neurons_body = neurons_body + ['PHSO1L', 'PHSO1R', 'AS10', 'DA7', 'VD11', 'VA11', 'PVPL', 'PVPR']

        
        data_body = [NeuroPALDataset.zscore(NeuroPALDataset.load_neuropal(files_body[i][:-14]+'.mat')[0].astype(float)) for i in range(len(files_body))]

        
        sz = np.array([d.shape for d in data_body]).max(0)

        for f in range(len(data_body)):
            pads = [(0,(sz[i]-data_body[f].shape[i]).astype(int)) for i in range(4)]
            data_body[f] = np.pad(data_body[f],pads)

        data_body = np.array(data_body).transpose([0,4,1,2,3])

        avg = nn.AvgPool3d(kernel_size=df[::-1], stride=df[::-1])

        neurons,col,pos,_ = utils.sort_mu(ims_body,neurons_body,min_counts=len(ims_body))
        self.positions = torch.tensor(pos)/scales.T[None,:,:]


        self.bounds = torch.tensor([(self.positions[:,i].min()-margin[i],self.positions[:,i].max()+margin[i]) for i in range(3)]).int()
        self.bounds[:,0] = torch.maximum(self.bounds[:,0],torch.zeros(3))
        self.bounds[:,1] = torch.minimum(self.bounds[:,1],torch.tensor(sz[:3].copy()))

        self.positions -= self.bounds[:,0][None,:,None]
        self.positions /= np.array(df).astype(np.float32)[:,None]

        data = data_body[:,:,self.bounds[0,0]:self.bounds[0,1],
                             self.bounds[1,0]:self.bounds[1,1],
                             self.bounds[2,0]:self.bounds[2,1]].astype(np.float32)
        
        self.shape = np.array(data.shape[::-1])
        self.shape[:3] = (self.bounds[:,1] - self.bounds[:,0]) // np.array(df)

        self.data = torch.tensor(data)
        self.data = avg(self.data)

        self.positions = torch.tensor(self.positions.permute(2,1,0)).float()
        self.neurons = neurons

    def __len__(self):
        '''Read out the size of the dataset.
        '''
        return self.data.shape[0]

    def __getitem__(self,idx):
        '''Read a batch of images and corresponding point clouds indexed by idx.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data[idx,:,:,:,:]
        sample[sample<0] = 0

        return sample,self.positions[:,:,idx]



class Neuron:
    """Properties and methods related to a single neuron"""
    def __init__(self):
        """Construct an instance of this class"""
         # Neuron position & color.
        self.position        = np.zeros((1,3))*np.nan # neuron pixel position (x,y,z)
        self.color           = np.zeros((1,4))*np.nan # neuron color based on fitting (R,G,B,W,...), W = white channel, values=[0-255]
        self.color_readout   = np.zeros((1,4))*np.nan # neuron color based on readout from the image
        self.baseline        = np.zeros((1,4))*np.nan # baseline noise values (R,G,B,W,...), values=[-1,1]
        self.covariance      = np.zeros((1,3,3))*np.nan # 3x3 covariance matrix that's fit to the neuron
        self.aligned         = np.zeros((7))*np.nan 
        
        # User neuron ID.
        self.annotation      = '' # neuron user selected annotation
        self.annotation_confidence = np.nan # user confidence about annotation
        
        # Auto neuron ID.
        self.deterministic_id    = None  # neuron ID assigned by the deterministic model
        self.probabilistic_ids   = None # neuron IDs listed by descending probability
        self.probabilistic_probs = None # neuron ID probabilities

class Image:
    """A list of neurons in a certain body part"""

    def __init__(self,bodypart,neurons=[],scale=np.ones((3))):
        """Construct an instance of this class"""
        self.bodypart = bodypart
        self.scale = scale
        self.neurons = neurons
        
    def get_positions(obj,scale=1):
        """Getter of neuron positions"""
        return np.array([neuron.position*scale for neuron in obj.neurons])
    
    def get_colors(obj):
        """Getter of neuron positions"""
        return np.array([neuron.color for neuron in obj.neurons])
    
    def get_colors_readout(obj):
        """Getter of neuron color readouts"""
        return np.array([neuron.color_readout for neuron in obj.neurons])
    
    def get_covariances(obj):
        """Getter of neuron covariances"""
        return np.array([neuron.covariance for neuron in obj.neurons])
    
    def get_aligned(obj):
        """Getter for aligned neurons"""
        return np.array([neuron.aligned for neuron in obj.neurons])
    
    def get_annotations(obj):
        """Getter of neuron annotations"""
        return [neuron.annotation for neuron in obj.neurons]