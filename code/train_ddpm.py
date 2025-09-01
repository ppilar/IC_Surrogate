# -*- coding: utf-8 -*-
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from ice._mod_conditional_denoising_diffusion_pytorch import cUnet, cGaussianDiffusion
from ice._mod_conditional_denoising_diffusion_pytorch_1d import cDataset1D, Trainer1D, cUnet1D, cGaussianDiffusion1D
from ice.utils import *
from ice.utils_ddpm import *
from ice.utils_eval import *

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    
    #%% initialize parameters
    pars = init_ds_pars('EM_grid')
    pars['batch_signals'] = False
    init_par(pars, 'run_dir', '../runs/runs_ddpm/')      
    init_par(pars, 'train_num_steps', 1000001)  
    init_par(pars, 'net_size', 3)
    init_par(pars, 'fixed_angle', True)
    init_par(pars, 'use_attn', False)
    
    
    #%% load data
    xn, yn, _, data_properties = load_data(pars['path_filtered'], shorten_seq=False, remove_invalid = pars['remove_invalid'], normalize = True)
    pars['Ns'], pars['Nt'], pars['NE'], pars['Ni0'], pars['seq_length'] = data_properties
    yn = yn[:,:2]
    
    pars['theta_ref'] = 0.74375
    if pars['fixed_angle'] == True:
        inds = np.where(yn[:,1] == pars['theta_ref'])[0]
        xn = xn[inds].repeat(50,0) #repeating 50x to avoid constant reshuffling of the dataset, which severely hampers performance; the minibatches remain random
        yn = yn[inds].repeat(50,0)
        pars['Nt'] = 1        
        pars['Ns'] = pars['NE']*pars['Ni0']*50
    
    data = (xn, yn)
    
    
    #%% train model    
    _, ddpm, trainer = get_model(data, pars, device)    
    trainer.train()
    
    
    #%% evaluate model
    del trainer, xn, yn, data
    ddpm.model.eval()
    ac = load_standard_ac(pars['seq_length'], device)
    with torch.no_grad():
        evaluate_theta_relation(ddpm, ac, pars['log_dir'], delta_t = 1)
        evaluate_theta_relation(ddpm, ac, pars['log_dir'], delta_t = 3)
        evaluate_theta_relation(ddpm, ac, pars['log_dir'], Nt = pars['sub_batch_size'], delta_t = 1)
        evaluate_theta_relation(ddpm, ac, pars['log_dir'], Nt = pars['sub_batch_size'], delta_t = 5)
        evaluate_theta_relation(ddpm, ac, pars['log_dir'], Nt = pars['sub_batch_size'], delta_t = 10)
        evaluate_ddpm(pars['log_dir'], eval_opt = 'samples')
        evaluate_ddpm(pars['log_dir'], eval_opt = 'dist')
    

