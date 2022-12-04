# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:09:22 2022

@author: Amin
"""
import networkx as nx

from torch.distributions import constraints

from sklearn.neighbors import kneighbors_graph

import torch.nn.functional as F
import torch.nn as nn

from pyro.infer import SVI, Trace_ELBO

import pyro.distributions as dist
import pyro

import numpy as np
import torch

import utils

# %%
class ImageFlowField(nn.Module):
    '''Class for learning atlases for image datasets using free form flow field
        transformations with regularization using point cloud positions.
    '''
    def __init__(self,sz,A,positions=None,centers=None,n_channels=4,device='cuda'):
        '''Class constructor

        Parameters
        ----------
        sz : TYPE
        A : TYPE
        positions : TYPE, optional
        centers : TYPE, optional
        n_channels : TYPE, optional
        device : TYPE, optional
        '''
        
        super(ImageFlowField, self).__init__()
        # Spatial transformer localization-network
        self.encoder = nn.Sequential(
            nn.Conv3d(n_channels, 3, kernel_size=[5,5,1]),
            nn.Conv3d(3, 3, kernel_size=[7,7,1]),
            nn.Conv3d(3, 3, kernel_size=[9,9,1]),
            nn.Conv3d(3, 3, kernel_size=[11,11,1]),
            nn.ReLU(True),
            nn.Conv3d(3, 3, kernel_size=[13,13,1]),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(3, 3, kernel_size=[13,13,1]),
            nn.ConvTranspose3d(3, 3, kernel_size=[11,11,1]),
            nn.ConvTranspose3d(3, 3, kernel_size=[9,9,1]),
            nn.ConvTranspose3d(3, 3, kernel_size=[7,7,1]),
            nn.ReLU(True),
            nn.ConvTranspose3d(3, 3, kernel_size=[5,5,1]),
            nn.Sigmoid()
        )
        
        self.A = A
        self.to(device)
        
        
        # start with identity transformation
        for p in self.parameters(): p.data.fill_(0)

        self.id_flow = torch.stack(torch.where(torch.ones((sz[0],sz[1],sz[2])))).T.reshape(sz[0],sz[1],sz[2],3).float().to(device)
        for d in range(self.id_flow.shape[3]):
            factor = sz[d]-1 if sz[d] > 1 else 1
            self.id_flow[:,:,:,d] = 2*self.id_flow[:,:,:,d]/factor - 1
        self.id_flow = self.id_flow[:,:,:,[2,1,0]]
        self.sz = torch.tensor(sz).float().to(device)
        self.P = positions.to(device).float()
        self.centers = centers.to(device).float()
        self.device = device
        
    def forward(self, x, times):
        z = self.encoder(x)
        grid = -1+2*self.decoder(z).permute([0,2,3,4,1])
        flow = grid+self.id_flow[None,:,:,:]
        flow[:,:,:,:,0] = 0
        X_t = F.grid_sample(x,flow)
        
        
        reg = torch.stack([ImageFlowField.regularizer(flow[t,:,:,:,[2,1,0]],self.sz,self.P[:,:,times[t]],self.centers) 
                           for t in range(grid.shape[0])])
        moved = []
        for t in range(grid.shape[0]):
            a = flow[t,:,:,:,[2,1,0]][self.P[:,:,times[t]].round().long().T.tolist()]
            moved.append(2*self.P[:,:,times[t]]-(.5+.5*a)*(self.sz-1))
        
        return X_t,grid,reg,moved
    
    @staticmethod
    def regularizer(grid,sz,P,Q):
        a = grid[P.round().long().T.tolist()]
        moved = 2*P-(.5+.5*a)*(sz-1)
        d_ = torch.cdist(moved.contiguous(),Q.contiguous()) 
        reg = d_.diag().mean()
        return reg
    
    
# %%
class ImagePiecewiseRigid(nn.Module):
    '''Class for learning atlases for image datasets using piecewise rigid (or rigid)
        transformations with regularization using point cloud positions.
    '''
    def __init__(
            self,sz,A,mesh=[2,2,1],tess=None,nbs=None,
            centers=None,positions=None,n_channels=4,std=10,device='cuda'
        ):
        super(ImagePiecewiseRigid, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv3d(n_channels, 8, kernel_size=[7,7,1]),
            nn.MaxPool3d([2,2,1], stride=[2,2,1]),
            nn.ReLU(True),
            nn.Conv3d(8, 10, kernel_size=[5,5,1]),
            nn.MaxPool3d([2,2,1], stride=[2,2,1]),
            nn.ReLU(True)
        )
        
        comp_sz = lambda x: (((x-6)//2)-4)//2
        
        self.device = device
        self.sz = torch.tensor(sz).float().to(device)
        self.center = (torch.tensor(self.sz)/2).float().to(device)
        self.numel = 10*comp_sz(sz[0])*comp_sz(sz[1])*sz[2]
        # Regressor for the 3 * 2 affine matrix
        
        n_rows,n_cols,n_secs = mesh
        tess = [np.array([(2*r+1)*sz[0]/(2*n_rows),(2*c+1)*sz[1]/(2*n_cols),(2*s+1)*sz[2]/(2*n_secs)]) 
                for r in range(n_rows) for c in range(n_cols) for s in range(n_secs)]
        if n_rows*n_cols*n_secs == 1: nbs = [[]]
        else:  nbs = [[i for i in nx.Graph(kneighbors_graph(np.array(tess),2)).neighbors(j)] for j in range(len(tess))]
        
        self.tess = torch.tensor([sz/2]) if tess is None else torch.tensor(tess)
        self.tess = self.tess.float().to(device)
        self.nbs = [[]] if nbs is None else nbs

        
        grid = torch.tensor(np.array(np.where(np.ones(list(sz))),dtype=float).T).float().to(device)
        
        self.grid = grid
        
        self.fc_loc = nn.Sequential(
            nn.Linear(self.numel, 32),
            nn.ReLU(True),
            nn.Linear(32, 6*len(self.tess)),
            nn.Tanh()
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()
        
        torch.pi = torch.acos(torch.zeros(1)).item()
        
        self.A = A.to(device)
        self.to(device)
        
        
        self.centers = centers.to(device).float() if centers is not None else None
        
        # precompute
        dist = torch.cdist(self.grid, self.tess)
        norm = lambda x: torch.exp(-x/std)/torch.exp(-x/std).sum(1)[:,None]
        self.dist = norm(dist)
        self.volumetric = False if self.A.shape[3] == 1 else True

        if positions is not None:
            self.positions = positions.permute(2,0,1).to(device)
            norm = lambda x: torch.exp(-x/std)/torch.exp(-x/std).sum(2)[...,None]
            dist_pc = torch.cdist(self.positions, self.tess[None])
            self.dist_pc = norm(dist_pc)
    
    def forward(self,x,idx):
        z = self.localization(x).view(-1, self.numel)
        T = self.fc_loc(z).view(x.shape[0],len(self.tess),6)
        r,t = self.rigid_t(torch.pi*T[:,:,:3], T[:,:,3:])
        
        grid = self.piecewise_flow(r,t)
        
        reg_ss = self.regularizer_ss(r,t,idx) if self.centers is not None else None
        reg_mm = self.regularizer(r,t)

        X_t = F.grid_sample(x,grid)
        
        return X_t,grid,reg_ss,reg_mm
    
    def regularizer(self,r,t,eps=.1):
        reg = torch.zeros(r.shape[0]).to(self.device)
        for i in range(len(self.tess)):
            for j in self.nbs[i]:
                Q = (self.tess[i]/2+self.tess[j]/2)[:,None] + torch.randn(3,2).to(self.device)
                Q_i = torch.einsum('bkt,kn->btn',r[:,i,:,:],Q) + t[:,i,:,None]
                Q_j = torch.einsum('bkt,kn->btn',r[:,j,:,:],Q) + t[:,j,:,None]
                reg += torch.norm(Q_i-Q_j,dim=2).mean(1)
            
        return reg
    
    def regularizer_ss(self,r,t,idx):
        pos = self.positions[idx]
        moved = ((torch.einsum('blk,btks->blts',self.centers[None],r)+t[:,None,:,:])*self.dist_pc[idx][...,None]).sum(2)
        reg = ((moved-pos)**2).sum(2).mean(1)
        return reg
    
    def piecewise_flow(self,r,t):
        flow = (torch.einsum('lk,btks->btls',self.grid,r)+t[:,:,None,:])*self.dist.T[None,:,:,None]
        sz = self.sz.clone()
        if sz[2] == 1: sz[2] = 2
        norm_flow = 2*flow.sum(1)/(sz[None,None,:]-1)-1
        return norm_flow.view((r.shape[0],self.sz[0].int().item(),self.sz[1].int().item(),self.sz[2].int().item(),3))[...,[2,1,0]]

    
    def rigid_t(self,alpha,trans):
        
        cos = torch.cos(alpha)
        sin = torch.sin(alpha)
        z = torch.zeros(alpha[:,:,0].shape).to(self.device)
        o = torch.ones(alpha[:,:,0].shape).to(self.device)
        
        b,t,_ = trans.shape
        
        X = torch.stack((cos[:,:,0],-sin[:,:,0],z,
                         sin[:,:,0], cos[:,:,0],z,
                         z,z,o)
                        ,2).view(b,t,3,3)
        
        Y = torch.stack((cos[:,:,1],z,sin[:,:,1],
                         z,o,z,
                         -sin[:,:,1],z,cos[:,:,1])
                        ,2).view(b,t,3,3)
        
        Z = torch.stack((o,z,z,
                         z,cos[:,:,2],-sin[:,:,2],
                         z,sin[:,:,2], cos[:,:,2])
                        ,2).view(b,t,3,3)
        
        XYZ = torch.einsum('btmn,btnk,btks->btms',X,Y,Z) if self.volumetric else X
        
        T = trans*self.sz[None,None,:]+\
            torch.einsum('m,bkmt->bkt',self.center,
                 torch.eye(3).to(self.device)[None,None,:,:]-XYZ)
            
        if not self.volumetric:  T[:,:,2] = 0
            
        return XYZ, T
    
    def det_jac(self,x,pos):
        z = self.localization(x).view(-1, self.numel)
        T = self.fc_loc(z)
        T = T.view(x.shape[0],len(self.tess),6)
        r,_ = self.rigid_t(torch.pi*T[:,:,:3], T[:,:,3:])
        jac = torch.einsum('nk,bkst->bnst',self.dist,r).det()
        return jac.view((jac.shape[0],self.sz[0].int().item(),self.sz[1].int().item(),self.sz[2].int().item()))

    def observation_loss(self,data):
        loss = ((data-self.A[None,...])**2).mean()
        # loss = -(data*self.A[None,...]).mean()
        return loss
        
    def estimate_theta(self,aligned):
        self.A = aligned.median(0)[0].detach()
        # self.A = aligned.mean(0).detach()
        
    def predict(self,testloader):
        test_loss,test_reg_ss = [],[]
        
        registered = np.zeros(list(testloader.dataset.shape))
        unregistered = np.zeros(list(testloader.dataset.shape))
        shape = list(testloader.dataset.shape)
        jac = np.zeros(shape[:3]+shape[4:])
        bs = testloader.batch_size
        
        for batch_idx, data in enumerate(testloader):
            with torch.no_grad():
                x_t,flow,reg_ss,reg_mm = self(data[0].to(self.device),data[1])
                test_loss.append(self.observation_loss(x_t).item())
                if reg_ss is not None: test_reg_ss.append(reg_ss.mean().item())
                
                x_t = x_t.detach().cpu().numpy()
                registered[:,:,:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(x_t,[2,3,4,1,0])
                unregistered[:,:,:,:,batch_idx*bs:batch_idx*bs+x_t.shape[0]] = np.transpose(data[0],[2,3,4,1,0])
                jac_ = self.det_jac(data[0].to(self.device),data[1])
                jac_ = jac_.detach().cpu().numpy()

                jac[:,:,:,batch_idx*bs:batch_idx*bs+jac_.shape[0]] = np.transpose(jac_,[1,2,3,0])
                

        return unregistered,registered,jac,test_loss,test_reg_ss
        
# %%
class ImageQuadratic(nn.Module):
    def __init__(
            self,sz,A,mesh=[2,2,1],centers=None,
            positions=None,n_channels=4,std=10,device='cuda'
        ):
        super().__init__()
        
        self.sz = torch.tensor(sz).float().to(device)
        
        flow_id = torch.cat(torch.where(torch.ones(list(sz)))).reshape([3]+list(sz)).permute([1,2,3,0]).float().to(device)
        flow_id = 2*flow_id/(self.sz-1)-1
        flow_id[torch.isnan(flow_id)] = 0
        
        self.transformed = utils.quadratic_basis(flow_id).to(device)
        
        self.betas = [
                torch.zeros(1,3)[:,:,None].repeat(1,1,positions.shape[2]).to(device),
                torch.zeros(1,3)[:,:,None].repeat(1,1,positions.shape[2]).to(device),
                torch.zeros(6,3)[:,:,None].repeat(1,1,positions.shape[2]).to(device)
            ]
        
        for i in range(len(self.betas)): self.betas[i].requires_grad = True
        
        self.device = device
        
        n_rows,n_cols,n_secs = [2,2,1] 
        tess = [np.array([(2*r+1)*sz[0]/(2*n_rows),(2*c+1)*sz[1]/(2*n_cols),(2*s+1)*sz[2]/(2*n_secs)]) 
                for r in range(n_rows) for c in range(n_cols) for s in range(n_secs)]
        self.tess = torch.tensor(tess).float().to(device)
        
        torch.pi = torch.acos(torch.zeros(1)).item()
        
        self.A = A.to(device)
        self.to(device)
        
        self.centers = centers.to(device).float() if centers is not None else None
        
        # precompute
        self.volumetric = False if self.A.shape[3] == 1 else True
        self.positions = positions.to(device)

    
    @property
    def beta(self):
        return torch.cat([
                    torch.tanh(self.betas[0]), 
                    self.rotation(torch.tanh(self.betas[1].permute(0,2,1))*torch.pi)[0], 
                    torch.tanh(self.betas[2])
                ],0)
        
    def parameters(self):
        return self.betas
    
    # Spatial transformer network forward function
    def forward(self,x,idx):
        grid = torch.einsum('mnza,abt->tmnzb', self.transformed, self.beta[:,:,idx])[...,[2,1,0]]
        if not self.volumetric: grid[:,:,:,:,2] = 0
        X_t = F.grid_sample(x,grid)

        reg_ss = self.regularizer_ss(idx)
        reg_mm = ((utils.quadratic_det_jac(self.beta[:,:,idx],self.tess.T)-1)**2).sum(1)
        
        return X_t,grid,reg_ss,reg_mm
    
    def regularizer_ss(self,idx):
        if self.positions is None or self.positions.shape[2] == 1: return None
        
        sz = self.sz.clone()
        if sz[2] == 1: sz[2] = 0
        
        P = -1+2*self.positions[:,:,idx]/(sz[None,:,None]-1)
        Q = -1+2*self.centers/(sz[None,:]-1)
        
        Q_q = utils.quadratic_basis(Q).to(self.device)
        # Q_q = torch.stack([utils.quadratic_basis(Q[:,:,t]).to(self.device) for t in range(len(idx))])
        moved = torch.einsum('abt,tna->nbt',self.beta[:,:,idx],Q_q[None,:,:])
        
        reg = ((moved-P)**2).sum(1).mean(0)
        return reg
    
    def rotation(self,alpha):
        cos = torch.cos(alpha)
        sin = torch.sin(alpha)
        z = torch.zeros(alpha[:,:,0].shape).to(self.device)
        o = torch.ones(alpha[:,:,0].shape).to(self.device)
        
        b,t,_ = alpha.shape
        
        X = torch.stack((cos[:,:,0],-sin[:,:,0],z,
                         sin[:,:,0], cos[:,:,0],z,
                         z,z,o)
                        ,2).view(b,t,3,3)
        
        Y = torch.stack((cos[:,:,1],z,sin[:,:,1],
                         z,o,z,
                         -sin[:,:,1],z,cos[:,:,1])
                        ,2).view(b,t,3,3)
        
        Z = torch.stack((o,z,z,
                         z,cos[:,:,2],-sin[:,:,2],
                         z,sin[:,:,2], cos[:,:,2])
                        ,2).view(b,t,3,3)
        
        XYZ = torch.einsum('btmn,btnk,btks->btms',X,Y,Z)
        
        return XYZ.permute(0,2,3,1)
    
    def observation_loss(self,data):
        loss = -(data*self.A[None,...]).mean()
        return loss

    def estimate_theta(self,aligned):
        self.A = aligned.mean(0).detach()

# %%
from abc import abstractmethod

class PCPiecewiseRigid(nn.Module):
    '''Class for learning atlases for point cloud datasets using piecewise rigid
        transformations with regularization using pairwise distances of CoM of pieces.
    '''
    def __init__(self,A,sz,tess=None,mask=None,device='cuda'):
        super(PCPiecewiseRigid, self).__init__()
        
        self.tess = [np.arange(A.shape[1])] if tess is None else tess
        self.mask = torch.zeros((0,0)).to(device) if mask is None else torch.tensor(mask).to(device) 
        
        # Localization-network
        self.localization = nn.Sequential(
            nn.Linear(A.shape[1]*3, 32),
            nn.ReLU(True),
            nn.Linear(32, 6*len(self.tess)),
            nn.Tanh()
        )

        
        self.color_transform = nn.Sequential(
            nn.Identity(),
            # nn.ReLU(),
            # nn.Softmax(dim=1)
        )
        
        self.device = device
        torch.pi = torch.acos(torch.zeros(1)).item()
        self.to(device)
        self.A = torch.tensor(A).float().to(device)
        
        self.center = torch.tensor(sz/2).float().to(device)
        self.sz = torch.tensor(sz).float().to(device)
        
        self.localization[2].weight.data.zero_()
        self.localization[2].bias.data.zero_()
        
        self.theta_c = self.A[3:,:].T
        self.theta_l = self.A[:3,:].T
        
        self.sigma_c = torch.ones((1)).to(device)
        self.sigma_l = torch.ones((1)).to(device)
        
        
    def regularizer(self, x):
        pts_1 = torch.stack([self.A[:3,t].mean(1) for t in self.tess],1).T.contiguous()
        pts_2 = torch.stack([x[:,:3,t].mean(2) for t in self.tess]).permute(1,0,2).contiguous()
        diff = torch.cdist(pts_1[None,:,:],pts_1[None,:,:]) - torch.cdist(pts_2,pts_2)
        return ((self.mask[None,:,:]*diff)**2).sum([1,2])
        
    def forward(self, x, args):
        X_t = torch.zeros(x.shape).to(self.device)
        z_l = self.localization(x[:,:3,:].view(x.shape[0],x.shape[2]*3))
        z_l = z_l.view(z_l.shape[0],len(self.tess),6)
        
        r_, t_ = self.rigid_t(torch.pi*z_l[:,:,3:],z_l[:,:,3:])
        for t in range(len(self.tess)):
            X_t[:,:3,self.tess[t]] = torch.einsum('bkt,bkn->btn',r_[:,t,:,:],x[:,:3,self.tess[t]]) + t_[:,t,:,None]
        
        z_c = self.color_transform(x[:,3:6,:])
        X_t[:,3:,:] = z_c
        
        reg = self.regularizer(X_t)
        return X_t,None,None,reg
    
    def rigid_t(self,alpha,trans):
        cos = torch.cos(alpha)
        sin = torch.sin(alpha)
        z = torch.zeros(alpha[:,:,0].shape).to(self.device)
        o = torch.ones(alpha[:,:,0].shape).to(self.device)
        
        b,t,_ = trans.shape
        
        X = torch.stack((cos[:,:,0],-sin[:,:,0],z,
                         sin[:,:,0], cos[:,:,0],z,
                         z,z,o)
                        ,2).view(b,t,3,3)
        
        Y = torch.stack((cos[:,:,1],z,sin[:,:,1],
                         z,o,z,
                         -sin[:,:,1],z,cos[:,:,1])
                        ,2).view(b,t,3,3)
        
        Z = torch.stack((o,z,z,
                         z,cos[:,:,2],-sin[:,:,2],
                         z,sin[:,:,2], cos[:,:,2])
                        ,2).view(b,t,3,3)
        
        XYZ = torch.einsum('btmn,btnk,btks->btms',X,Y,Z)
        
        T = trans*self.sz[None,None,:]+\
            torch.einsum('m,bkmt->bkt',self.center,
                 torch.eye(3).to(self.device)[None,None,:,:]-XYZ)
        return XYZ, T
    
    @abstractmethod
    def observation_loss(self):
        pass
    
    @abstractmethod
    def prior(self):
        pass
    
    @abstractmethod
    def estimate_theta(self):
        pass
    
    def predict(self,testloader):
        aligned = torch.zeros((len(testloader.dataset),6,self.A.shape[1]))
        bs = testloader.batch_size
        
        test_loss,test_reg_ss = [],[]
        for batch_idx, data in enumerate(testloader):
            with torch.no_grad():
                x_t,_,reg_ss,reg_mm = self(data[0].to(self.device),data[1])
                
                test_reg_ss.append(((x_t[:,:3]-self.A[:3])**2).sum(1).mean().item())
                aligned[batch_idx*bs:batch_idx*bs+x_t.shape[0],:,:] = x_t.detach().cpu()
                test_loss.append(self.observation_loss(x_t).item())
            
        return aligned, test_loss, test_reg_ss
        

# %%

class PCPiecewiseRigidDirichlet(PCPiecewiseRigid):
    '''Prior atlas distribution over position is multivariate normal while the 
        prior over the color is dirichlet.
    '''
    def __init__(self,A,sz,tess=None,mask=None,device='cuda',tau=1):
        super(PCPiecewiseRigidDirichlet, self).__init__(A,sz,tess=None,mask=None,device='cuda')
        self.tau=tau
        self.theta_c = torch.softmax(self.A[3:,:].T/tau,dim=1)
        
    def forward(self, x, args):
        X_t = torch.zeros(x.shape).to(self.device)
        z_l = self.localization(x[:,:3,:].view(x.shape[0],x.shape[2]*3))
        z_l = z_l.view(z_l.shape[0],len(self.tess),6)
        
        r_, t_ = self.rigid_t(torch.pi*z_l[:,:,3:],z_l[:,:,3:])
        for t in range(len(self.tess)):
            X_t[:,:3,self.tess[t]] = torch.einsum('bkt,bkn->btn',r_[:,t,:,:],x[:,:3,self.tess[t]]) + t_[:,t,:,None]
        
        z_c = self.color_transform(x[:,3:6,:])
        z_c = torch.softmax(z_c/self.tau,dim=1)
        X_t[:,3:,:] = z_c
        
        reg = self.regularizer(X_t)
        return X_t,None,reg,None


    def observation_loss(self,data,gamma=.01):
        # Computing negative log likelihood as the cost
        a = data.permute(0,2,1)
        with pyro.plate('neurons',a.size(1)):
            with pyro.plate('data',a.size(0)):
                loss_c = -dist.Dirichlet(self.theta_c).log_prob(a[:,:,3:])
                loss_l = -dist.MultivariateNormal(self.theta_l,self.sigma_l*torch.eye(3).to(self.device)).log_prob(a[:,:,:3])
        return loss_l.mean()+gamma*loss_c.mean()
        
    def prior(self,data):
        theta_c = pyro.param('theta_c', self.theta_c,constraint=constraints.simplex)
        theta_l = pyro.param('theta_l', self.theta_l)
        sigma_l = pyro.param('sigma_l', self.sigma_l, constraint=constraints.positive)
        
        with pyro.plate('neurons',data.size(1)):
            with pyro.plate('data',data.size(0)):
                pyro.sample('obs_c', dist.Dirichlet(theta_c),obs=data[:,:,3:])
                pyro.sample('obs_l', dist.MultivariateNormal(theta_l,sigma_l*torch.eye(3).to(self.device)),obs=data[:,:,:3])
    
    def estimate_theta(self,aligned,lr=.1,thresh=.1,patience=100):
        a = aligned.permute(0,2,1)
        
        def my_lr(module_name, param_name):
            return {"lr": lr/100.0} if param_name == 'theta_c' else {"lr": lr}

        guide = pyro.infer.autoguide.AutoDelta(self.prior)
        pyro.clear_param_store()
        optimizer = pyro.optim.Adam(my_lr)
        svi = SVI(self.prior, guide, optimizer, loss=Trace_ELBO())
        
        loss,counter = 0,0
        while counter < patience:
            new_loss = svi.step(a)
            counter = counter+1 if abs(loss-new_loss) < thresh else 0
            loss = new_loss
        
        with torch.no_grad():
            self.theta_c = pyro.param('theta_c')
            self.theta_l = pyro.param('theta_l')
            self.sigma_l = pyro.param('sigma_l')
            
            self.A = torch.cat((self.theta_l,self.theta_c),1).T
        


class PCPiecewiseRigidNormal(PCPiecewiseRigid):
    '''Class for learning atlases for point cloud datasets using piecewise rigid
        (or rigid) transformations with regularization using pairwise distances 
        of CoM of pieces. Prior atlas distribution over position and color is 
        multivariate normal.
    '''
    
    def observation_loss(self,data,gamma=.01):
        # Computing negative log likelihood as the cost
        a = data.permute(0,2,1)
        with pyro.plate('neurons',a.size(1)):
            with pyro.plate('data',a.size(0)):
                loss_c = -dist.MultivariateNormal(self.theta_c,self.sigma_c*torch.eye(3).to(self.device)).log_prob(a[:,:,3:])
                loss_l = -dist.MultivariateNormal(self.theta_l,self.sigma_l*torch.eye(3).to(self.device)).log_prob(a[:,:,:3])
        return loss_l.mean()+gamma*loss_c.mean()
        
    def prior(self,data):
        theta_c = pyro.param('theta_c', self.theta_c)
        theta_l = pyro.param('theta_l', self.theta_l)
        
        sigma_c = pyro.param('sigma_c', self.sigma_c, constraint=constraints.positive)
        sigma_l = pyro.param('sigma_l', self.sigma_l, constraint=constraints.positive)
        
        with pyro.plate('neurons',data.size(1)):
            with pyro.plate('data',data.size(0)):
                pyro.sample('obs_c', dist.MultivariateNormal(theta_c,sigma_c*torch.eye(3).to(self.device)),obs=data[:,:,3:])
                pyro.sample('obs_l', dist.MultivariateNormal(theta_l,sigma_l*torch.eye(3).to(self.device)),obs=data[:,:,:3])
    
    def estimate_theta(self,aligned,lr=.1,thresh=.1,patience=100):
        a = aligned.permute(0,2,1)
        
        def my_lr(module_name, param_name):
            return {"lr": lr/100.0} if param_name == 'theta_c' else {"lr": lr}

        guide = pyro.infer.autoguide.AutoDelta(self.prior)
        pyro.clear_param_store()
        optimizer = pyro.optim.Adam(my_lr)
        svi = SVI(self.prior, guide, optimizer, loss=Trace_ELBO())
        
        loss,counter = 0,0
        while counter < patience:
            new_loss = svi.step(a)
            counter = counter+1 if abs(loss-new_loss) < thresh else 0
            loss = new_loss
        
        
        with torch.no_grad():
            self.theta_c = pyro.param('theta_c')
            self.theta_l = pyro.param('theta_l')
            
            self.sigma_c = pyro.param('sigma_c')
            self.sigma_l = pyro.param('sigma_l')
            
            self.A = torch.cat((self.theta_l,self.theta_c),1).T


# %%
def train_model(
        model,dataloader,optimizer,
        gamma_re=1,gamma_ss=0,gamma_mm=0,
        epochs=20,epochs_theta=10,
        device='cuda',
        save=False,file=None
    ):
    '''Training one of the model instances using batches of data.

    Parameters
    ----------
    model (nn.Module): and instance of the model to be trained (check specific instances).
    dataloader (torch.utils.data.Dataset): pytorch dataloader class (check datasets.py).
    optimizer (torch.optim): pytorch optimizer module (normally we use Adam).
    gamma (float): regularization (different for each model, check specific instances).
    epochs (int): number of epochs overall for training atlas.
    epochs_theta (int): number of epochs after which theta is updated.
    device (string): 'cpu' or 'cuda'.

    Returns
    -------
    losses (list): training losses through iterations.
    '''
    
    losses = []
    atlases = []
    
    for epoch in range(1,epochs+1):
        print('Epoch ' + str(epoch))
        model.train()
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            X_t,_,reg_ss,reg_mm = model(data[0].to(device),data[1].to(device))
            recon = model.observation_loss(X_t)
            
            loss = gamma_re*recon
            if reg_ss is not None: loss += gamma_ss*reg_ss.mean()
            if reg_mm is not None: loss += gamma_mm*reg_mm.mean()
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print('Recon: ' + str(recon))
                print('SS Regularization: ' + str(reg_ss))
                print('MM Regularization: ' + str(reg_mm))
        
        if (epoch+1) % epochs_theta == 0:
            with torch.no_grad():
                model.estimate_theta(X_t)
                atlases.append(model.A.detach().cpu().numpy())
        
    if save:
        torch.save(model.state_dict(), file)

    return losses, atlases


