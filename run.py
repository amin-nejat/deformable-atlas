# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 20:12:40 2022

@author: Amin
"""

import sys
sys.path.insert(1, '../codes')

import glob

import datasets
import models
import visualization

from torch.utils.data import DataLoader
import torch.optim as optim

import argparse
import yaml

import numpy as np
import os
# %%
def get_args():
    '''Parsing the arguments when this file is called from console
    '''
    parser = argparse.ArgumentParser(description='Runner for CSLDS',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', '-c', metavar='Configuration',help='Configuration file address',default='/')
    parser.add_argument('--output', '-o', metavar='Output',help='Folder to save the output and results',default='/')
    
    return parser.parse_args()
# %%
if __name__ == '__main__':
    '''Read the arguments and run given the configuration in args
    '''
    args = get_args()
    if not os.path.exists(args.output): os.makedirs(args.output)
    with open(args.config, 'r') as stream: config = yaml.safe_load(stream)
    
    train_params = config['train_params']
    if 'folder' in train_params.keys():
        files = glob.glob(train_params['folder']+'*_STRUCT.mat')
        train_params['files'] = files
        train_params.pop('folder')
    
    dataset = eval('datasets.'+config['dataset'])(
        **train_params
    )
    trainloader = DataLoader(dataset, batch_size=config['bs'], shuffle=False, num_workers=0)
    
    batch,pos = next(iter(trainloader))
    
    # %%
    model_params = config['model_params']
    if config['model'][:2] == 'PC':
        model = eval('models.'+config['model'])(
            A=dataset.A,sz=dataset.sz,
            tess=dataset.tessellation,
            mask=dataset.mask,
            **model_params
        )


    if config['model'][:5] == 'Image':
        model = eval('models.'+config['model'])(
            sz=dataset.shape[:3],
            A=dataset.A,
            centers=dataset.a_positions,
            positions=dataset.positions,
            **model_params
        )
    
    optimization_params = config['optimization_params']
    optimizer = optim.Adam(model.parameters(), lr=optimization_params['lr'] )
    optimization_params.pop('lr')
    
    train_loss = models.train_model(
        model,trainloader,optimizer,
        save=True,file=args.output+'model.torch',
        **optimization_params
    )
    
    # %%
    test_params = config['test_params']
    if 'folder' in test_params.keys():
        files = glob.glob(test_params['folder']+'*_STRUCT.mat')
        test_params['files'] = files
        test_params.pop('folder')
        
    testdataset = eval('datasets.'+config['dataset'])(
        **test_params
    )
    testloader = DataLoader(testdataset , batch_size=1, shuffle=False, num_workers=0)
    prediction = model.predict(testloader)
    
    # %%
    if 'visualize_image' in config['visualizations']:
        unregistered,registered,jac,test_loss,test_reg_ss = prediction
        np.savetxt(args.output+'loss.txt',[train_loss,test_loss])
        
        visualization.visualize_image(
            registered.transpose(0,1,2,3,4)[:,:,:,:3].mean(4),
            titlestr='Registered',
            scale=1,
            microns=.2,
            p=dataset.a_positions[:,[0,1,2]].numpy(),
            c=dataset.a_positions.numpy()*0,
            names=dataset.neurons,
            olp=False,
            save=True,file=args.output+'registered'
        )
        
        visualization.visualize_image(
            unregistered.transpose(0,1,2,3,4)[:,:,:,:3].mean(4),
            titlestr='Unregistered',
            scale=1,
            microns=.2,
            p=dataset.a_positions[:,[0,1,2]].numpy(),
            c=dataset.a_positions.numpy()*0,
            names=dataset.neurons,
            olp=False,
            save=True,file=args.output+'unregistered'
        )
        
    if 'visualize_pc' in config['visualizations']:
        visualization.visualize_pc(
            dataset.point_cloud,
            save=True,file=args.output+'unregistered'
        )

        test_aligned, _, test_loss = prediction
        train_aligned, _, train_loss = model.predict(trainloader)
        
        visualization.visualize_pc(
            train_aligned,atlas=model.A.T.detach().cpu().numpy(),
            names=testdataset.neurons,olp=True,
            title_str='Inferred Atlas',
            save=True,file=args.output+'atlas'
        )
        
        visualization.visualize_pc(
            test_aligned,atlas=model.A.T.detach().cpu().numpy(),
            names=testdataset.neurons,olp=True,three_d=True,
            title_str='Atlas and Test Data',
            save=True,file=args.output+'registered'
        )
        
        np.savetxt(args.output+'loss.txt',[train_loss,test_loss])
