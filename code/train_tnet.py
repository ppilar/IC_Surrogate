# -*- coding: utf-8 -*-
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

from ice.utils import init_random_seeds, init_par
from ice.utils_tnet import *
from ice.utils_ddpm import init_ds_pars
from ice.utils_eval import evaluate_theta_relation, evaluate_theta_relation_detailed, load_standard_ac, get_ppath



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

#%% load data
init_random_seeds(s=1)
pars = init_ds_pars('EM_grid')
xn, yn, _, data_properties = load_data(pars['path_filtered'], shorten_seq=False, remove_invalid = pars['remove_invalid'], normalize = True)
init_standard_pars_tnet(pars, data_properties)

#%% set model parameters
pars['device'] = device 
pars['itges'] = 300000

print('train tnet!')
#%% train with different architectures
mopt_vec = ['UNet']#['CNN', 'Block', 'UNet']
for jm, mopt in enumerate(mopt_vec):
    if jm == 0:
        pars['Block_layer_opt'] = 'standard'
    #if jm == 0:
    #    pars['Block_layer_opt'] = 'double'
        
    pars['model_opt'] = mopt    
    TNet = ThetaNet(pars)
    TNet.train(xn, yn)
        
    #%% continue training
    #TNet.train(xn, yn, 500, device)
    
    #%% make comparison plot for given energy and theta range
    ppath,_ = get_ppath(TNet.writer.log_dir, '')
    for E in [0., 0.2, 0.4, 0.6, 0.8]:
        theta_eval(TNet, xn, yn, bs=10, Nt=9, delta_t=5, theta_0=55.82, E=E, ppath = ppath)
     
    
    ac = load_standard_ac(xn.shape[1], device)
    #%%
    Evec = [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.]
    t0_vec = [0.5]
    with torch.no_grad():
        evaluate_theta_relation(TNet, ac, ppath, E_opt = '', bs = 10, Evec = Evec, t0_vec = t0_vec, Nt = 13, delta_t = 3, model_opt='tnet', plot_comment='tnet')
        evaluate_theta_relation_detailed(TNet, (xn, yn), ppath)


