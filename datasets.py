# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:53:52 2022

@author: Amin
"""
from torch.utils.data import Dataset

import torch.nn as nn
import torch

from scipy.io import loadmat

import numpy as np
import scipy as sp

import utils
import h5py

# import pandas
from PIL import Image
import glob

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
class FlyWingDataset(Dataset):
    '''Image dataset, to store and read image data in batches.
    '''
    def __init__(self,folder,sex,genotype,side,device='cuda',margin=[100,100],df=[4,4]):
        
        file_name = genotype+'_'+sex+'_'+side
        files = glob.glob(folder+file_name+'_lei_2X*')
        
        sz = np.array(np.array(Image.open(files[0])).shape[:2])

        self.files = files
        self.avg = nn.AvgPool2d(kernel_size=df, stride=df)
        self.bounds = np.array([
            [margin[0],sz[0]-margin[0]],
            [margin[1],sz[1]-margin[1]]
        ])
        self.A = self.process(0)
        

    def __len__(self):
        '''Read out the size of the dataset.
        '''
        return len(self.files)-1
    
    def process(self,idx):
        image = np.array(Image.open(self.files[idx])).astype(float)
        if len(image.shape) == 2: 
            image = torch.tensor(image[:,:,None][:,:,[0,0]]).float()
        else:
            image = 1-torch.tensor(image).float()/255
            
        image = image[self.bounds[0][0]:self.bounds[0][1],
                      self.bounds[1][0]:self.bounds[1][1],1]
        image = self.avg(image[None,:,:])
        image[image < 0] = 0
        
        return image[...,None]
    
    def __getitem__(self,idx):
        '''Read a batch indexed by idx.
        '''
        
        idx = idx+1
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.process(idx)

        return image,idx
        
        
        
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

# %%
class NeuroPALPC(Dataset):
    def __init__(self,info,files,body,neurons=None):
        data = NeuroPALDataset.load_celegans_data(info)
        ims = [NeuroPALDataset.load_neuropal_id(file) for file in files]
        
        ims_body = [im for im in ims if im.bodypart == body]
        scales = np.array([im.scale for im in ims_body])
        
        if body == 'head':
            self.neurons = list(set([neuron for ganglion in data['ganglia'][:9] for neuron in ganglion['neurons']]))
            self.neurons += ['AMSOL', 'AMSOR']
            
            self.ganglia = [list(set([neuron for neuron in ganglion['neurons']])) for ganglion in data['ganglia'][:9]]
            self.ganglia += [['AMSOR','AMSOL']]

        if body == 'tail':
            self.neurons = list(set([neuron for ganglion in data['ganglia'][13:21] for neuron in ganglion['neurons']]))
            self.neurons += ['PHSO1L', 'PHSO1R', 'AS10', 'DA7', 'VD11', 'VA11', 'PVPL', 'PVPR']
            
            self.ganglia = [list(set([neuron for neuron in ganglion['neurons']])) for ganglion in data['ganglia'][13:]]
            self.ganglia += [['AS10', 'DA7', 'VD11','VA11']]
            self.ganglia += [['PHSO1R','PHSO1L']]

        if neurons is None:
            self.neurons,col,pos,_ = utils.sort_mu(ims_body,self.neurons,min_counts=len(ims_body))
        else:
            self.neurons,col,pos,_ = utils.sort_mu(ims_body,neurons,min_counts=0)
            
        pos = pos*scales.T[None,:,:]
        
        tess = [[self.neurons.index(n) for n in ganglion if n in self.neurons] for ganglion in self.ganglia]
        
        # self.tessellation = tess
        if body == 'head': 
            self.tessellation = [tess[0]+tess[1]+tess[2]+tess[3]+tess[4],tess[4]+tess[5]+tess[6]+tess[7],tess[7]+tess[8]]
        if body == 'tail': 
            self.tessellation = [tess[4]+tess[0],tess[0]+tess[1],tess[1]+tess[2]+tess[3]+tess[5]]
        
        self.mask = 1-np.eye(len(self.tessellation))
        # self.point_cloud = np.hstack((pos,sp.special.softmax(col[:,:3,:],axis=1))).transpose([2,1,0])
        
        self.point_cloud = np.hstack((pos,col[:,:3,:])).transpose([2,1,0])
        
        self.A = self.point_cloud[0]
        
        self.sz = torch.tensor([pos[:,i,:].max() for i in range(3)])
        
    def __len__(self):
        '''Read out the size of the dataset.
        '''
        return self.point_cloud.shape[0]
  
    def __getitem__(self,idx):
        '''Read a batch of images and corresponding point clouds indexed by idx.
        '''
        return torch.tensor(self.point_cloud[idx]).float(),idx
    
# %%
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
        im = NeuroPALImage(bodypart=content['bodypart'].lower(),scale=content['scale'])
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
        content = h5py.File(file)
        scale = np.array(content['info_scale']).T
        rgbw = np.array(content['info_RGBW']).squeeze().astype(int)
        
        image = content['data']
        return image, scale, rgbw 

    @staticmethod
    def load_celegans_data(file):
        content = loadmat(file,simplify_cells=True)
        return content

    def __init__(
            self,info,files,body,neurons=None,
            df=[1,1,1],margin=[10,10,3],
            atlas=None,atlas_order=[1,0,2]
        ):
        
        data = NeuroPALDataset.load_celegans_data(info)
        ims = [NeuroPALDataset.load_neuropal_id(file) for file in files]
        
        ims_body = [im for im in ims if im.bodypart == body]
        files_body = [files[i] for i in range(len(files)) if ims[i].bodypart == body]
        scales = np.array([im.scale for im in ims_body])


        if body == 'head':
            self.neurons = list(set([neuron for ganglion in data['ganglia'][:9] for neuron in ganglion['neurons']]))
            self.neurons += ['AMSOL', 'AMSOR']

        if body == 'tail':
            self.neurons = list(set([neuron for ganglion in data['ganglia'][13:21] for neuron in ganglion['neurons']]))
            self.neurons += ['PHSO1L', 'PHSO1R', 'AS10', 'DA7', 'VD11', 'VA11', 'PVPL', 'PVPR']
        
        
        if neurons is None:
        	self.neurons,col,pos,_ = utils.sort_mu(ims_body,self.neurons,min_counts=len(ims_body))
        else:
        	self.neurons,col,pos,_ = utils.sort_mu(ims_body,neurons,min_counts=len(ims_body))
            

        contents = [NeuroPALDataset.load_neuropal(files_body[i][:-14]+'.h5') for i in range(len(files_body))]
        self.data_body = [a[0] for a in contents]
        self.rgbws = [a[2] for a in contents]
        
        sz = np.array([d.shape for d in self.data_body]).max(0)[::-1]
        
        if atlas is not None:
            atlas = NeuroPALDataset.load_celegans_data(atlas)['atlas']
            neurons_atlas = atlas[body]['N'].tolist()
            idx = [neurons_atlas.index(n) for n in self.neurons]
            a_positions = atlas[body]['model']['mu'][idx,atlas_order]/ims_body[0].scale

        self.sz = sz
        self.pads = []
        for f in range(len(self.data_body)):
            pads = [(0,(sz[i]-self.data_body[f].shape[::-1][i]).astype(int)) for i in range(4)]
            self.pads.append(pads)

        self.avg = nn.AvgPool3d(kernel_size=df, stride=df)
        
        positions = torch.tensor(pos)/scales.T[None,:,:]
        
        # self.bounds = torch.tensor([(positions[:,i].min()-margin[i],positions[:,i].max()+margin[i]) for i in range(3)]).int()
        self.bounds = torch.tensor([(0,sz[i]) for i in range(3)])
        self.bounds[:,0] = torch.maximum(self.bounds[:,0],torch.zeros(3))
        self.bounds[:,1] = torch.minimum(self.bounds[:,1],torch.tensor(sz[:3].copy()))

        positions -= self.bounds[:,0][None,:,None]
        positions /= np.array(df).astype(np.float32)[:,None]
        positions = positions.float()
        
        self.sz[:3] = np.diff(self.bounds).squeeze()
        self.sz[:3] = self.sz[:3]/np.array(df)
        self.sz[3] = len(self.rgbws[0])
        self.positions = positions[:,:,1:]
        self.shape = self.sz.tolist() + [len(self.data_body)-1]
        
        self.A = self.process(0)
        
        if atlas is None: 
            self.a_positions = positions[:,:,0]
        else:
            sz_ = np.array(self.A.shape[1:])
            a_positions /= np.array(df).astype(np.float32)
            a_positions = a_positions - a_positions.mean(0) + sz_/2
            self.a_positions = torch.tensor(a_positions).float()
            
    def process(self,idx):
        data = NeuroPALDataset.zscore(
            np.array(self.data_body[idx])[self.rgbws[idx]-1]
        ).astype(np.float32).transpose([3,2,1,0])
        data = np.pad(data,self.pads[idx])
        
        data = data[self.bounds[0,0]:self.bounds[0,1],
                    self.bounds[1,0]:self.bounds[1,1],
                    self.bounds[2,0]:self.bounds[2,1],:
                 ]
        
        data = torch.tensor(data).float().permute(3,0,1,2)
        data[data<0] = 0
        data = self.avg(data)
        return data
        
    def __len__(self):
        '''Read out the size of the dataset.
        '''
        return self.shape[-1]

    def __getitem__(self,idx):
        '''Read a batch of images and corresponding point clouds indexed by idx.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.process(idx+1)

        return sample,idx



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

class NeuroPALImage:
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
    

    