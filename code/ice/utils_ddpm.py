# -*- coding: utf-8 -*-
import torch
import numpy as np
import pickle

from ._mod_conditional_denoising_diffusion_pytorch import cUnet, cGaussianDiffusion
from ._mod_conditional_denoising_diffusion_pytorch_1d import cDataset1D, Trainer1D, cUnet1D, cGaussianDiffusion1D
from .utils import init_par
from .utils_ac import complete_reverse_an
from .utils_load import get_data_path #get_path_filtered




def init_ds_pars(ds_opt = 'EM_grid', pars=None):
    if pars is None: pars = dict()
    init_par(pars, 'ds_opt', ds_opt)
    
    pars['sub_batch_size'] = 2
    pars['batch_signals'] = False#True
    if pars['ds_opt'] == 'EM_grid':
        pars['path_filtered'] = get_data_path(lx=1024)       
        
    if pars['ds_opt'] == 'EM_grid_lowE':
        pars['path_filtered'] = get_data_path(lx=896)
        
        
    ###
        
    #if pars['batch_signals'] == True:
    pars['remove_invalid'] = False
    init_par(pars, 'use_feature_conditioning', 0)
    init_par(pars, 'use_cross_attention', 0)
        
    return pars
        

def get_model(dataset, pars, device):
    from .utils import init_par
    
    pars['device'] = device
    init_par(pars, 'objective', 'pred_v')
    init_par(pars, 'time_steps', 50) #500
    init_par(pars, 'sampling_steps', 20)    #50 
    init_par(pars, 'train_num_steps', 1000001)
    init_par(pars, 'gradient_accumulate_every', 2)
    init_par(pars, 'ema_decay', 0.995)    
    init_par(pars, 'lr', 3e-4)#8e-5)
    
    
    init_par(pars, 'net_size', 1)
    
    if pars['net_size'] == 0:
        init_par(pars, 'Unet_dim', 64)
        init_par(pars, 'Unet_dim_mults', (1, 2))
    if pars['net_size'] == 1:
        init_par(pars, 'Unet_dim', 64)
        init_par(pars, 'Unet_dim_mults', (1, 2, 4, 8))
    if pars['net_size'] == 2:
        init_par(pars, 'Unet_dim', 64)
        init_par(pars, 'Unet_dim_mults', (1, 2, 4, 8, 16, 16))
    if pars['net_size'] == 3:
        init_par(pars, 'Unet_dim', 32)
        init_par(pars, 'Unet_dim_mults', (1, 2, 4, 8, 16, 32))
    if pars['net_size'] == 4:
        init_par(pars, 'Unet_dim', 32)
        init_par(pars, 'Unet_dim_mults', (1, 2, 4, 8, 16, 32, 64))    
    if pars['net_size'] == 5:
        init_par(pars, 'Unet_dim', 8)
        init_par(pars, 'Unet_dim_mults', (1, 2, 4, 8, 16, 32))
        
    
    
    init_par(pars, 'sub_batch_size', 2)
    if pars['batch_signals'] == False:
        init_par(pars, 'bs', 128)
    else:
        init_par(pars, 'bs', 64)       
    
    
    init_par(pars, 'use_cross_attention', False)
    init_par(pars, 'use_feature_conditioning', False)
    
    
    UNet = cUnet1D(
        dim = pars['Unet_dim'],
        dim_mults = pars['Unet_dim_mults'],
        channels = 1,
        pars = pars
    ).to(device)
    

    ddpm = cGaussianDiffusion1D(
        model=UNet,
        seq_length = pars['seq_length'],
        timesteps = pars['time_steps'],
        sampling_timesteps = pars['sampling_steps'],
        objective = pars['objective'],
        Nc = -1,
        batch_signals = pars['batch_signals'],
        sub_batch_size = pars['sub_batch_size']
    ).to(device)
    
    pars['Unet_layers'] = len(pars['Unet_dim_mults'])
    trainer = Trainer1D(
        ddpm,
        dataset = dataset,
        train_batch_size = pars['bs'],
        train_lr = pars['lr'],
        train_num_steps = pars['train_num_steps'],         # total training steps
        gradient_accumulate_every = pars['gradient_accumulate_every'],    # gradient accumulation steps
        ema_decay = pars['ema_decay'],                # exponential moving average decay
        amp = True,               # turn on mixed precision
        pars = pars
    )
        
    return UNet, ddpm, trainer



def sample_from_ddpm(ddpm, N, bs = 256, classes = False, epsilon = False):
    device = ddpm.device
    print(device)
    sges = torch.zeros(N, ddpm.model.pars['seq_length']).to(device).float() #samples
    cges = torch.zeros(N, 2).to(device).float() #classes   
    
        
    if N < bs:
        bs = N
        
    if type(epsilon) is not bool:
        bs0 = int(epsilon.shape[0]/ddpm.n_batch_signals)
        bs = bs0*ddpm.n_batch_signals
        
    
    for i in range(int(np.ceil(N/bs))):  
        if type(classes) is bool:
            buf = torch.rand(bs,2, device=device).float()
        else:
            buf = classes[i*bs:(i+1)*bs,:]
            
        if type(epsilon) is bool:
            eps_buf = epsilon
        else:
            eps_buf = epsilon[i*bs:(i+1)*bs,:,:]
            
        sbuf, cbuf = ddpm.sample(bs, buf, eps_buf);
        
        imax = min((i+1)*bs, N)
        imax2 = bs - ((i+1)*bs - N) if (i+1)*bs > N else bs
        sges[i*bs:imax,:] = sbuf.squeeze()[:imax2,:]
        cges[i*bs:imax,:] = cbuf[:imax2,:]     
        
        
    return sges, cges


