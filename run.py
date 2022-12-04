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

import torch
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
    args.output += os.path.split(args.config)[1][:-5]+'/'
    if not os.path.exists(args.output): os.makedirs(args.output)
    with open(args.config, 'r') as stream: config = yaml.safe_load(stream)
    
    train_params = config['train_params']
    if config['dataset'] == 'NeuroPALDataset':
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
    
    train_loss,atlases = models.train_model(
        model,trainloader,optimizer,
        save=True,file=args.output+'model.torch',
        **optimization_params
    )
    
    # train_prediction = model.predict(trainloader)
    
    # %%
    test_params = config['test_params']
    if config['dataset'] == 'NeuroPALDataset':
        files = glob.glob(test_params['folder']+'*_STRUCT.mat')
        test_params['files'] = files
        test_params.pop('folder')
        
    testdataset = eval('datasets.'+config['dataset'])(
        **test_params
    )
    
    if config['model'][:2] == 'PC':
        test_model = eval('models.'+config['model'])(
            A=testdataset.A,sz=testdataset.sz,
            tess=testdataset.tessellation,
            mask=testdataset.mask,
            **model_params
        )


    if config['model'][:5] == 'Image':
        test_model = eval('models.'+config['model'])(
            sz=testdataset.shape[:3],
            A=testdataset.A,
            centers=testdataset.a_positions,
            positions=testdataset.positions,
            **model_params
        )

    test_model.load_state_dict(torch.load(args.output+'model.torch'))


    testloader = DataLoader(testdataset , batch_size=1, shuffle=False, num_workers=0)
    test_prediction = test_model.predict(testloader)
    
    # %%
    if 'visualize_image' in config['visualizations']:
        unregistered,registered,jac,test_loss,test_reg_ss = test_prediction
        save_loss = test_reg_ss if len(test_reg_ss) > 0 else test_loss
        np.savetxt(args.output+'loss.txt',[save_loss])
        
        visualization.visualize_image(
            registered.transpose(0,1,2,3,4)[:,:,:,:3].mean(4),
            titlestr='Registered',
            factor=2,
            scale=1,
            microns=.2,
            p=dataset.a_positions[:,[0,1,2]].numpy() if dataset.a_positions is not None else None,
            c=dataset.a_positions.numpy()*0 if dataset.a_positions is not None else None,
            names=dataset.neurons if hasattr(dataset, 'neurons') else None,
            olp=False,
            save=True,file=args.output+'registered'
        )
        
        visualization.visualize_image(
            unregistered.transpose(0,1,2,3,4)[:,:,:,:3].mean(4),
            titlestr='Unregistered',
            factor=2,
            scale=1,
            microns=.2,
            p=dataset.a_positions[:,[0,1,2]].numpy() if dataset.a_positions is not None else None,
            c=dataset.a_positions.numpy()*0 if dataset.a_positions is not None else None,
            names=dataset.neurons if hasattr(dataset, 'neurons') else None,
            olp=False,
            save=True,file=args.output+'unregistered'
        )
        
        visualization.visualize_subjects(
            [a.transpose(1,2,3,0) for a in atlases], ['Iter ' + str(i) for i in range(len(atlases))],
            save=True,file=args.output+'atlases'
        )

        
    if 'visualize_pc' in config['visualizations']:
        visualization.visualize_pc(
            dataset.point_cloud,
            save=True,file=args.output+'unregistered'
        )

        test_aligned, _, test_loss = test_prediction
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
