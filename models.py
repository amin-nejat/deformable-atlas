# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:09:22 2022

@author: Amin
"""

from torch.distributions import constraints
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import utils as U
import torch
import pyro

# %%
class ImageFlowField(nn.Module):
    def __init__(self,sz,A,positions=None,centers=None,n_channels=4,device='cuda'):
        super(ImageFlowField, self).__init__()
        # Spatial transformer localization-network
        self.encoder = nn.Sequential(
            nn.Conv3d(n_channels, 3, kernel_size=[5,5,1]),
            # nn.Conv3d(3, 3, kernel_size=[7,7,1]),
            # nn.Conv3d(3, 3, kernel_size=[9,9,1]),
            # nn.Conv3d(3, 3, kernel_size=[11,11,1]),
            nn.ReLU(True),
            nn.Conv3d(3, 3, kernel_size=[13,13,1]),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(3, 3, kernel_size=[13,13,1]),
            # nn.ConvTranspose3d(3, 3, kernel_size=[11,11,1]),
            # nn.ConvTranspose3d(3, 3, kernel_size=[9,9,1]),
            # nn.ConvTranspose3d(3, 3, kernel_size=[7,7,1]),
            nn.ReLU(True),
            nn.ConvTranspose3d(3, 3, kernel_size=[5,5,1]),
            nn.Sigmoid()
        )
        
        self.A = A
        self.to(device)
        
        # for p in self.parameters():
        #     p.data.fill_(0)

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
        d_ = torch.cdist(moved,Q) #-1+2*Q/sz
        reg = d_.diag().mean()
        # d  = torch.cdist(2*P/sz,2*P/sz)
        # reg = ((d_ - d)**2).sum()
        
        return reg
    
    
# %%
class ImagePiecewiseRigid(nn.Module):
    def __init__(self,sz,A,positions=None,centers=None,n_channels=4,std=10,device='cuda'):
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
        
        
        # n_rows,n_cols,n_secs = [1,2,1]
        n_rows,n_cols,n_secs = [2,2,1]
        tess = [np.array([(2*r+1)*sz[0]/(2*n_rows),
                          (2*c+1)*sz[1]/(2*n_cols),
                          (2*s+1)*sz[2]/(2*n_secs)
               ]) for r in range(n_rows) for c in range(n_cols) for s in range(n_secs)]
        tess = torch.tensor(tess).float().to(device)
        # self.nbs = [[i] for i in range(1,n_cols*n_rows*n_secs)]+[[]]
        # self.nbs = [[1],[2],[5],[4],[3],[6],[7],[8],[]]
        self.nbs = [[1],[3],[2],[]]
        # self.nbs = [[1],[]]
        # self.nbs = [[]]

        grid = torch.tensor(np.array(np.where(np.ones(list(sz))),dtype=float).T).float().to(device)
        
        self.tess = tess
        self.grid = grid
        
        self.fc_loc = nn.Sequential(
            nn.Linear(self.numel, 32),
            nn.ReLU(True),
            nn.Linear(32, 6*len(tess)),
            nn.Tanh()
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()
        
        torch.pi = torch.acos(torch.zeros(1)).item()
        
        self.A = A
        self.to(device)
        
        
        self.P = positions.to(device).float() if positions is not None else None
        self.centers = centers.to(device).float() if centers is not None else None
        
        
        # precompute
        dist = torch.cdist(grid, tess)
        norm = lambda x: torch.exp(-x/std)/torch.exp(-x/std).sum(1)[:,None]
        self.dist = norm(dist)
        
    
    
    def forward(self, x, times):
        z = self.localization(x).view(-1, self.numel)
        T = self.fc_loc(z)
        T = T.view(x.shape[0],len(self.tess),6)
        
        r,t = self.rigid_t(torch.pi*T[:,:,:3], T[:,:,3:])
        grid = self.piecewise_flow(r,t)[:,:,:,:,[2,1,0]]
        # grid[:,:,:,:,0] = 0
        
        if self.P is not None:
            reg = torch.stack([ImageFlowField.regularizer(grid[t,:,:,:,[2,1,0]],
                         self.sz,self.P[:,:,times[t]],self.centers) for t in range(grid.shape[0])])
        else:
            reg = None
            
        X_t = F.grid_sample(x,grid)
        
        moved = []
        if self.P is not None:
            for t in range(grid.shape[0]):
                a = grid[t,:,:,:,[2,1,0]][self.P[:,:,times[t]].round().long().T.tolist()]
                moved.append(2*self.P[:,:,times[t]]-(.5+.5*a)*(self.sz-1))
            
        return X_t,grid,reg,moved
        
    @staticmethod
    def regularizer_pr(r, t, tess, nbs, eps=.1, device='cuda'):
        reg = torch.tensor(0.0).to(device)
        for i in range(len(tess)):
            for j in nbs[i]:
                Q_ = (tess[i]+tess[j])/2
                a = (tess[i][0] - tess[j][0])/(tess[j][1] - tess[i][1]+1e-6)
                Q = torch.stack((torch.tensor([Q_[0]+eps, eps*a+Q_[1], Q_[2]+eps]),
                                 torch.tensor([Q_[0]-eps,-eps*a+Q_[1], Q_[2]-eps]))).to(device)
                reg += ((Q@r[i]+t[i]-
                        (Q@r[j]+t[j]))**2).mean()
            
        return reg
    
    @staticmethod
    def regularizer(grid,sz,P,Q):
        a = grid[P.round().long().T.tolist()]
        moved = 2*P-(.5+.5*a)*(sz-1)
        d_ = torch.cdist(moved,Q)
        reg = d_.diag().mean()
        
        return reg
    
    
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

        
    def piecewise_flow(self,r,t):
        flow = (torch.einsum('lk,btks->btls',self.grid,r)+t[:,:,None,:])*self.dist.T[None,:,:,None]
        norm_flow = 2*flow.sum(1)/self.sz[None,None,:]-1
        return norm_flow.view((r.shape[0],self.sz[0].int().item(),self.sz[1].int().item(),self.sz[2].int().item(),3))
    
    
    def observation_loss(self,data):
        loss = F.mse_loss(data,self.A[None,...])
        return loss
        # loss = -(data*self.A[None,...]).mean()
        
    def estimate_theta(self,aligned):
        self.A = aligned.mean(0)
        
            

# %%
class PCPiecewiseRigid(nn.Module):
    def __init__(self,A,sz,device='cuda'):
        super(PCPiecewiseRigid, self).__init__()
        # Spatial transformer localization-network
        self.tess = [np.arange(20),np.arange(20,40),np.arange(40,A.shape[1])]
        # self.nbs = [[1],[]]


        self.localization = nn.Sequential(
            nn.Linear(A.shape[1]*3, 32),
            nn.ReLU(True),
            nn.Linear(32, 6*len(self.tess)),
            nn.Tanh()
        )

        
        self.color_transform = nn.Sequential(
            # nn.Linear(A.shape[1]*3, 1),
            nn.Softmax(dim=1)
        )
        
        self.device = device
        torch.pi = torch.acos(torch.zeros(1)).item()
        self.to(device)
        self.A = torch.tensor(A).float().to(device)
        
        self.center = torch.tensor(sz/2).float().to(device)
        self.sz = torch.tensor(sz).float().to(device)
        
        self.localization[2].weight.data.zero_()
        self.localization[2].bias.data.zero_()
        
        self.theta_c = torch.ones((A.shape[1],3)).to(device)
        self.theta_l = torch.ones((A.shape[1],3)).to(device)
        
    def regularizer(self, x):
        return torch.norm(torch.cdist(self.A[:3,:].T,self.A[:3,:].T)[None,:,:]-
                          torch.cdist(x[:,:3,:].permute(0,2,1),x[:,:3,:].permute(0,2,1)),dim=[1,2])
        
    def forward(self, x, times):
        X_t = torch.zeros(x.shape).to(self.device)
        z_l = self.localization(x[:,:3,:].reshape(x.shape[0],x.shape[2]*3))
        z_l = z_l.view(z_l.shape[0],len(self.tess),6)
        
        r_, t_ = self.rigid_t(torch.pi*z_l[:,:,3:],z_l[:,:,3:])
        for t in range(len(self.tess)):
            X_t[:,:3,self.tess[t]] = torch.einsum('bkt,bkn->btn',r_[:,t,:,:],x[:,:3,self.tess[t]]) + t_[:,t,:,None]
        
        X_t[:,3:,:] = self.color_transform(x[:,3:6,:])
        
        reg = self.regularizer(x)
        return X_t,None,reg,None
    
    def observation_loss(self,data,gamma=.1):
        # Computing negative log likelihood as the cost
        a = data.permute(0,2,1)
        with pyro.plate('neurons',a.size(1)):
            with pyro.plate('data',a.size(0)):
                loss_l = -dist.Dirichlet(self.theta_c).log_prob(a[:,:,3:])
                loss_c = -dist.MultivariateNormal(self.theta_l,torch.eye(3).to(self.device)).log_prob(a[:,:,:3])
        
        return loss_l.mean()+gamma*loss_c.mean()
        
    def prior(self,data):
        theta_c = pyro.param('theta_c', self.theta_c)
        theta_l = pyro.param('theta_l', self.theta_l)
        with pyro.plate('neurons',data.size(1)):
            with pyro.plate('data',data.size(0)):
                pyro.sample('obs_c', dist.Dirichlet(theta_c),obs=data[:,:,3:])
                pyro.sample('obs_l', dist.MultivariateNormal(theta_l,torch.eye(3).to(self.device)),obs=data[:,:,:3])
    
    def estimate_theta(self,aligned,lr=1.,thresh=10,patience=10):
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
            
        self.theta_c = pyro.param('theta_c')
        self.theta_l = pyro.param('theta_l')
        
        self.A = torch.cat((self.theta_l,self.theta_c),1).T
        
        
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



# %%
def train_model(model,dataloader,optimizer,gamma=0,epochs=20,epochs_theta=10,device='cuda'):
    losses = []
    
    for epoch in range(1,epochs+1):
        print('Epoch ' + str(epoch))
        model.train()
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            X_t,_,reg,_ = model(data[0].to(device),data[1])
            recon = model.observation_loss(X_t)
            loss = recon+gamma*reg.mean() if reg is not None else recon
            loss.backward()
            optimizer.step()
            
            losses.append(loss)
            
            if batch_idx % 10 == 0:
                print('Recon: ' + str(recon))
                print('Reg: ' + str(reg))
        
        if (epoch+1) % epochs_theta == 0:
            model.estimate_theta(X_t.detach())
        
    return losses
    