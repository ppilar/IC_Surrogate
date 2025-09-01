# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from functools import partial

from ._mod_conditional_denoising_diffusion_pytorch_1d import get_shuffle_indices, cDataset1D, cycle
from .utils import *
from .utils_vary_theta import plot_vary_theta_comparison, get_NuRadioMC_parameters, get_vary_theta_pars
from .utils_interpolate import *
#from .nets_theta_transformation import *
from .utils_load import *


#set parameters that have not been initialized before to standard values
def init_standard_pars_tnet(pars, data_properties):
    from .utils import init_par
    pars['Ns'], pars['Nt'], pars['NE'], pars['Ni0'], pars['seq_length'] = data_properties
    
    pars['theta_ref'] = 0.74375 #0.875#9.8750e-01
    pars['sub_batch_size'] = 1
    pars['batch_signals'] = True
    pars['batch_size'] = 128
    pars['lr'] = 3e-4#1e-3
    pars['itges'] = 200000#250000
    pars['fsched'] = 0.5
    pars['it_sched'] = 750000
    pars['train_val_split'] = True#False
    pars['weight_decay'] = 0#5e-4
    pars['fdropout'] = 0
    pars['transform_signal'] = True
    pars['use_attn'] = False
    pars['use_block2'] = True
    
    pars['extract_features'] = True
    pars['feature_dim'] = 32
    pars['Unet_dim'] = 8


def get_loss(xtrue, xpred):
    return ((xtrue - xpred)**2).sum(1)



def transform_signal(xges, t0, t1):
    if type(xges) == np.ndarray:
        xges = torch.tensor(xges).float()        
        
    x1ges = torch.zeros(xges.shape, device = xges.device)
    
    
    if type(t0) in [float, np.float32]:
        t0 = torch.tensor([t0], device = xges.device).float()        
    if type(t1) in [float, np.float32]:
        t1 = torch.tensor([t1], device = xges.device).float()        
    if type(t0) == np.ndarray:
        t0 = torch.tensor(t0, device = xges.device).float()
    if type(t1) == np.ndarray:
        t1 = torch.tensor(t1, device = xges.device).float()  
    tc = 0.5 #corresponds to Cherenkov angle
    #a = 200
    
    
    #case 1:
    mask = ((t0<tc)&(t1<tc)) | ((t0>=tc)&(t1>=tc))
    buf = torch.where(mask.unsqueeze(1).repeat(1,xges.shape[-1]), torch.ones_like(xges), torch.zeros_like(xges))
    x1ges = x1ges + xges*buf
    
    #case 2:
    mask1 = ((t0<tc)&(t1>=tc)) | ((t0>=tc)&(t1<tc))
    buf = torch.where(mask1.unsqueeze(1).repeat(1,xges.shape[-1]), torch.ones_like(xges), torch.zeros_like(xges))
    x1buf = torch.flip(xges*buf, [1])
    x1buf = ((x1buf - 0.5)*(-1) + 0.5)*buf
    
    adjust = 1
    if not adjust:
        x1ges = x1ges + torch.roll(x1buf, shifts = [0, 25], dims = [0,1])
        f = torch.maximum(torch.abs(tc-t1)/torch.abs(tc-t0), torch.tensor(0.1, device=xges.device))
    else:
        x1ges = x1ges + torch.roll(x1buf, shifts = [0, 0], dims = [0,1])
        #f0 = 0.75 if t1 < tc else 1
        #off = 0
        f0 = 1 - 0.25*(t1 < tc)
        f = f0*torch.maximum(torch.abs(tc-t1)/torch.abs(tc-t0), torch.tensor(0.1, device=xges.device)) #0.01
    
    tvec0 = torch.linspace(0, xges.shape[1], xges.shape[1], device = xges.device)
    if xges.shape[1] == 896:
        tmid = 450
    elif xges.shape[1] == 1024:
        tmid = 512
    elif xges.shape[1] == 4992: #5000:
        tmid = 2496
    
    if not adjust:
        tvec = (tvec0.unsqueeze(0)-tmid)*f.unsqueeze(1) + tmid
    else:
        s = torch.sign(tvec0.unsqueeze(0)-tmid)
        dt = (tvec0.unsqueeze(0)-tmid).abs()
        #e = 1.003
        #f2 = dt/dt**(dt**e/dt)
        f2 = 1#torch.cos(dt/tmid*torch.pi/4)**2
        tvec = s*dt*f2*f.unsqueeze(1) + tmid
    
    
    x1ges_final = interp_vmap(x1ges-0.5, 1/(f*f2)).squeeze(1) + 0.5 #different interpolation function for vmap compatibility
    
    # x1ges_final = torch.zeros(xges.shape, device = xges.device)    
    # for j in range(xges.shape[0]):
    #     xbuf = interp(tvec[j], x1ges[j], tvec0)
    #     buf = torch.zeros(xges.shape, device = xges.device)
    #     buf[j] = 1
    #     x1ges_final = x1ges_final + xbuf*buf
            
        
    return x1ges_final#.squeeze()





def reshuffle_dataset(data, pars, Ni0_val = 0):
    if pars['batch_signals'] == True:
        print('reshuffling dataset!')
        print('')
        Ni0 = pars['Ni0'] - Ni0_val
        s = get_shuffle_indices(pars['Nt'], pars['NE'], Ni0)
        lx = data[0].shape[-1]
        buf0 = data[0][s].reshape((-1,pars['sub_batch_size'],lx))
        buf1 = data[1][s].reshape((-1,pars['sub_batch_size'],3))
        buf = cDataset1D((buf0, buf1))
    else:
        buf = data
    
    dl = DataLoader(buf, batch_size = pars['batch_size'], shuffle = True)
    dl = cycle(dl)
    return dl


def get_irefs(cbuf, yrefs):
    if type(cbuf) is not torch.Tensor:
        cbuf = torch.Tensor(cbuf).float()
    if type(yrefs) is not torch.Tensor:
        yrefs = torch.Tensor(yrefs).float()
    
    irefs = []
    if cbuf.ndim == 2:
        cbuf = cbuf.unsqueeze(1)
    for i in range(cbuf.shape[0]):    
        b1 = ((yrefs[:,0] - cbuf[i,0,0]).abs() < 1e-3).int()
        b2 = ((yrefs[:,2] - cbuf[i,0,2]).abs() < 1e-3).int()
        iref = torch.where(b1+b2==2)
        irefs.append(iref[0].item())
        
    return irefs


def compare_true_and_preprocessed_signals(xn, yn, i0=0, E0=0.1125, itheta0=0):
    inds = np.abs(yn[:,0] - E0) <= 5e-4
    ynE = yn[inds]
    xnE = xn[inds]
    buf = xnE[::10,:]
    
    x = xnE[161*i0:161*(i0+1),:] #TODO: use Nt instead of 161
    y = ynE[161*i0:161*(i0+1)]
    inds = np.argsort(y[:,1])
    x = x[inds]
    y = y[inds]      
    
    ivec = [0,20,40,60,75,120,140,155]
    for i1 in ivec:
        plot_true_and_preprocessed(x[itheta0], x[i1], y[itheta0,1], y[i1,1])


def get_ref_inds(theta_ref, yn):
    buf = yn.cpu() if type(yn) == torch.Tensor else yn    
    inds = np.where(np.abs(buf[:,1] - theta_ref) < 1e-3)[0]
    return inds

def get_ref_signals(theta_ref, xn, yn):
    inds = get_ref_inds(theta_ref, yn)
    xrefs = xn[inds]
    yrefs = yn[inds]
    return xrefs, yrefs


def find_closest_match(yn, E, t, i0=0):
    y = np.array([[E, t, i0]])
    i = np.argmin(np.abs(yn-y).sum(1))
    return yn[i], i

def get_xy_ref(Etvals, xrefs, yrefs):
    irefs = get_irefs(Etvals, yrefs)
    xref = xrefs[irefs]
    yref = yrefs[irefs]
    return xref, yref


def theta_eval0(TNet, xn, yn, bs=10, Nt=11, delta_t=1, theta_0=55.82, E=0.1, i0=0, ppath = '../plots/test/'):
    TNet.model.to('cpu')
    xrefs, yrefs = get_ref_signals(TNet.pars['theta_ref'], xn, yn)
    
    _, i = find_closest_match(yn, E, normalize_t(theta_0), i0=i0)
    print('closest i:', i)
    
    ibuf = int(Nt/2)
    cbuf = yn[i-ibuf*delta_t:i+ibuf*delta_t+1:delta_t]
    xbuf = xn[i-ibuf*delta_t:i+ibuf*delta_t+1:delta_t]
    
    
    if type(xbuf) != 'torch.Tensor':
        xbuf = torch.tensor(xbuf).float()
    if type(cbuf) != 'torch.Tensor':
        cbuf = torch.tensor(cbuf).float()
    
    xref, yref = get_xy_ref(cbuf, xrefs, yrefs)
    
    cbuf = cbuf.float()
    
    xref = torch.tensor(xref).float()
    yref = torch.tensor(yref).float()
    xpred, xbase = TNet.model((xref, yref), cbuf[:,0], cbuf[:,1])
    tvec, tlabel = get_tvec(xbase.shape[-1])
    
    ix0, ix1 = 300, 723
    for i in range(1):#Nt):
        plt.figure(figsize=(4.5,4))
        plt.plot(tvec[ix0:ix1], xbuf[i].cpu()[ix0:ix1]*2-1., color='C0', label='true')
        plt.plot(tvec[ix0:ix1], xref[i][ix0:ix1]*2-1., color='C3', label='gen')
        plt.plot(tvec[ix0:ix1], xbase[i].cpu().detach()[ix0:ix1]*2-1., linestyle='--', color='C1', label='pre')
        plt.plot(tvec[ix0:ix1], xpred[i].cpu().detach()[ix0:ix1]*2-1., linestyle='--', color='C2', label=r'$\theta$-Net')#'tnet')
        plt.title(get_Et_title(E,cbuf[i,1]))
        plt.legend()
        plt.xlabel(tlabel)
        plt.ylabel('normalized amplitude')
        
        plt.savefig(ppath + 's'+str(i)+ '_E%.2f'%(E)+'.pdf', bbox_inches='tight')


def theta_eval(TNet, xn, yn, bs=10, Nt=11, delta_t=1, theta_0=55.82, E=0.1, ppath = '../plots/test/', plot_comment=''):
    TNet.model.to('cpu')
    xrefs, yrefs = get_ref_signals(TNet.pars['theta_ref'], xn, yn)

    Et = get_vary_theta_pars(bs = bs, Nt = Nt, E = E, theta_0 = theta_0, delta_t = delta_t, device = 'cpu')
    buf = torch.linspace(0,bs-1,bs).repeat(Nt)
    Et = torch.cat((Et, buf.unsqueeze(1)),1)
    
    xref, yref = get_xy_ref(Et, xrefs, yrefs)    
    xref = torch.tensor(xref).float()
    yref = torch.tensor(yref).float()
    xpred, xbase = TNet.model((xref, yref), Et[:,0], Et[:,1])

    Nsp = 10
    ip = 10
    iN = [1]*bs
    parameters_scaled = get_NuRadioMC_parameters(Et.cpu().numpy())
    signals_filtered = 0.5*torch.ones(xpred.shape)
    
    plot_vary_theta_comparison(xpred.detach(), signals_filtered, parameters_scaled, bs, Nsp, iN[:ip], ppath=ppath, normalize_plot=0, plot_comment=plot_comment)





class ThetaNet(object):
    def __init__(self, pars):
        super().__init__()
        self.pars = pars
        self.device = pars['device']
        self.initialize_model(pars)
        self.it = 0
        self.ival = None
        
    def initialize_model(self, pars):
        from .nets_theta_transformation import cUnet1D_theta
        if pars['model_opt'] == 'UNet':
            init_par(pars, 'Unet_dim', 32)        
            init_par(pars, 'Unet_dim_mults', (1, 2, 4, 8, 16, 32))
            #init_par(pars, 'Unet_dim_mults', (1, 2, 4, 8, 16))
            self.model = cUnet1D_theta(pars, pars['Unet_dim'],
                                  dim_mults = pars['Unet_dim_mults'],
                                  channels = 1, 
                                  fdropout = pars['fdropout']).to(pars['device'])
        elif pars['model_opt'] == 'Block':
            init_par(pars, 'Block_layer_opt', 'standard')
            init_par(pars, 'Block_c0_dim', 32)
            self.model = BlockCNN(c0_dim = pars['Block_c0_dim'], layer_opt = pars['Block_layer_opt']).to(pars['device'])
        elif pars['model_opt'] == 'CNN':
            self.model = SimpleCNN().to(pars['device'])
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=pars['lr'], weight_decay=pars['weight_decay'])        
        
    def save(self, path, name='final'):
        if path[-1] != '/':
            path += '/'
        self.writer.close()
        data = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'writer': self.writer,
            'pars': self.pars,
            'it': self.it,
            'ival': self.ival
            }
        torch.save(data, path + name + '.pt')
        
    def load(self, path, device='cpu'):
        self.device = device #TODO: more elegant way?
        data = torch.load(path, map_location=device)        
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.writer = data['writer']
        self.pars = data['pars']
        self.it = data['it']
        if 'ival' in data:
            self.ival = data['ival']
        self.device = device
        
        
    def init_writer(self):    
        log_dir = '../runs/runs_theta_transformation/'
        if not hasattr(self, 'writer'):
            self.writer = initialize_writer(log_dir, comment0 = "", comment = self.pars['model_opt'])
        print_parameters(self.pars, self.writer.log_dir)
        save_parameters(self.pars, self.writer.log_dir)
        
    
    
    def evaluate_validation_loss(self, data_val, inds=False):
        self.model.eval()
        xrefs_val, yrefs_val = get_ref_signals(self.pars['theta_ref'], data_val[0], data_val[1])
        if type(inds) is not bool:
            data_buf = (data_val[0][inds][::10,:], data_val[1][inds][::10,:])
        else:
            data_buf = (data_val[0][::10,:], data_val[1][::10,:])
        
        N = data_buf[0].shape[0]
        bs = 128
        loss0_val = 0
        loss_val = 0
        with torch.no_grad():
            for j in range(int(np.ceil(N/bs))):
                (xbuf, cbuf) = (data_buf[0][j*bs:(j+1)*bs,:], data_buf[1][j*bs:(j+1)*bs,:])    
                xref, yref = get_xy_ref(cbuf, xrefs_val, yrefs_val)  
                
                
                cbuf = cbuf.to(self.device)
                xref = xref.to(self.device)
                yref = yref.to(self.device)
                xpred, xbase = self.model((xref, yref), cbuf[:,0], cbuf[:,1])          
                xpred = xpred.to('cpu')
                xbase = xbase.to('cpu')
                cbuf = cbuf.to('cpu')
                
                loss0_val += get_loss(xbuf, xbase).sum()/N
                loss_val += get_loss(xbuf, xpred).sum()/N
        
        self.model.train()        
        return loss0_val, loss_val
    
    
        
        
    def train_validation_split(self, data0):
        if self.ival is None:
            #omit certain iN from data
            self.iN_val = 0
            self.Ni0_val = 1
            self.ival = torch.where(data0[1][:,2] == self.iN_val)[0]
            self.Nval = self.ival.shape[0]
        
        inds_val, inds_train = get_bool_array(data0[0].shape[0], self.ival)
        data_train = (data0[0][inds_train==1], data0[1][inds_train==1])
        data_val = (data0[0][inds_val==1], data0[1][inds_val==1])

        return data_train, data_val
    
        
    def train(self, xn, yn, add_its = 0, device=''):
        self.pars['itges'] += add_its
        if device != '':
            self.device = device
        self.model.to(self.device)
        self.model.train()
        
        
        data = (torch.tensor(xn).float(), torch.tensor(yn).float())
        if self.pars['train_val_split']:
            data_train, data_val = self.train_validation_split(data)
        else:
            data_train = data
            data_val = -1
            self.Ni0_val = 0
        
        
        xrefs, yrefs = get_ref_signals(self.pars['theta_ref'], data_train[0], data_train[1])
        dl = reshuffle_dataset(data_train, self.pars, self.Ni0_val)        
        self.init_writer()
        scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.pars['fsched'])
        
        
        loss0_ges = []
        loss_ges = []
        it_per_epoch = int(xn.shape[0]/(self.pars['batch_size']*self.pars['sub_batch_size']))
        loop_obj = tqdm(range(self.it, self.pars['itges']))
        for i in loop_obj:
            self.optimizer.zero_grad()
            xbuf, cbuf = next(dl)            
            xbuf = xbuf.reshape(-1,xbuf.shape[2])
            cbuf = cbuf.reshape(-1,3)
            xref, yref = get_xy_ref(cbuf, xrefs, yrefs)
            
            xbuf = xbuf.to(self.device)
            xref = xref.to(self.device)
            yref = yref.to(self.device)
            cbuf = cbuf.to(self.device)
            
            
            xpred, xbase = self.model((xref, yref), cbuf[:,0], cbuf[:,1])
            
            
            loss0 = get_loss(xbuf, xbase).mean()
            loss = get_loss(xbuf, xpred).mean()
            loss.backward()
            self.optimizer.step()        
            
            
            ## evaluation
            
            loop_obj.set_description(f"Loss: {np.round(loss.item(),4)}")
            if self.it%20 == 0:
                self.writer.add_scalar("loss_baseline", loss0, self.it)
                self.writer.add_scalar("loss", loss, self.it)
                self.writer.add_scalar("loss_diff", loss - loss0, self.it)
            loss_ges.append(loss.item())
            loss0_ges.append(loss0.item())
                
            self.it += 1
            if self.it%self.pars['it_sched'] == 0 and self.it > 0:
                scheduler.step()
            if self.it%it_per_epoch == 0 and self.it > 0:
                dl = reshuffle_dataset(data_train, self.pars, self.Ni0_val)
            if self.it%50 == 0 and self.it>0 and type(data_val) is not int:
                loss0_val, loss_val = self.evaluate_validation_loss(data_val)                
                self.writer.add_scalar("val_loss_baseline", loss0_val, self.it)
                self.writer.add_scalar("val_loss", loss_val, self.it)
                
                inds_lowE = torch.where(data_val[1][:,0] <= 0.5)[0]
                inds_highE = torch.where(data_val[1][:,0] > 0.5)[0]
                inds_lowE_lowt = torch.where((data_val[1][:,0] <= 0.5) & (data_val[1][:,1] <= 0.5))[0]
                inds_lowE_hight = torch.where((data_val[1][:,0] <= 0.5) & (data_val[1][:,1] > 0.5))[0]
                inds_highE_lowt = torch.where((data_val[1][:,0] > 0.5) & (data_val[1][:,1] <= 0.5))[0]
                inds_highE_hight = torch.where((data_val[1][:,0] > 0.5) & (data_val[1][:,1] > 0.5))[0]
                self.writer.add_scalar("val_loss_lowE", self.evaluate_validation_loss(data_val, inds_lowE)[1], self.it)
                self.writer.add_scalar("val_loss_highE", self.evaluate_validation_loss(data_val, inds_highE)[1], self.it)
                self.writer.add_scalar("val_loss_lowE_lowt", self.evaluate_validation_loss(data_val, inds_lowE_lowt)[1], self.it)
                self.writer.add_scalar("val_loss_lowE_hight", self.evaluate_validation_loss(data_val, inds_lowE_hight)[1], self.it)
                self.writer.add_scalar("val_loss_highE_lowt", self.evaluate_validation_loss(data_val, inds_highE_lowt)[1], self.it)
                self.writer.add_scalar("val_loss_highE_hight", self.evaluate_validation_loss(data_val, inds_highE_hight)[1], self.it)
                
             
                
        self.model.eval()
        self.save(self.writer.log_dir)