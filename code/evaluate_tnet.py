# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt

from ice.utils import load_data
from ice.utils_tnet import theta_eval, theta_eval0
from ice.utils_ddpm import init_ds_pars
from ice.utils_eval import evaluate_theta_relation, load_models_for_evaluation, evaluate_theta_relation_detailed
from ice.utils_load import get_standard_tnet_path

#%% load data
pars = init_ds_pars()
xn, yn, a0, _ = load_data(pars['path_filtered'], shorten_seq=False, remove_invalid = pars['remove_invalid'], normalize = True)

#%% load net
model_path = '../runs/runs_theta_transformation/ref_Jun19_00-43-42_UNet'


if not 'model_path' in locals():
    model_path = get_standard_tnet_path(pars['ds_opt']) 
    
tnet, ac, ppath = load_models_for_evaluation(model_path, 'final', 'tnet', seq_length=xn.shape[1], device='cpu')


#%% eval 0: plot same signal for different theta values
for E in np.linspace(0.8,1.,11):
    theta_eval0(tnet, xn, yn, bs=2, Nt=15, delta_t=10, theta_0=55.82, E=E, i0=0, ppath = ppath)

#%% eval 1: plot same signal for different theta values
theta_eval(tnet, xn, yn, bs=2, Nt=11, delta_t=1, theta_0=65.82, E=0.1, ppath = ppath, plot_comment='tnet0')
theta_eval(tnet, xn, yn, bs=2, Nt=9, delta_t=4, theta_0=55.82, E=0.1, ppath = ppath, plot_comment='tnet0')

#%% eval 2: plot same signal for different theta values and compare to real signals (requires NuRadioMC)
Evec = [0., 0.1, 0.25, 0.5, 0.75, 0.9, 1.]
Evec = [0.9]
t0_vec = [0.5]
evaluate_theta_relation(tnet, ac, ppath, E_opt = '', bs = 10, Evec = Evec, t0_vec = t0_vec, Nt = 13, delta_t = 3, model_opt='tnet', plot_comment='tnet')

    
#%% evaluate absolute errors between signals and predictions on E-t grid
evaluate_theta_relation_detailed(tnet, (xn, yn), ppath)


