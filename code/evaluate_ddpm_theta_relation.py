#-*- coding: utf-8 -*-
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys


# from ice.utils_vary_theta import *
from ice.utils_eval import evaluate_theta_relation
from ice.utils_load import load_models_for_evaluation


if len(sys.argv) > 1:
    name_ddpm = sys.argv[1]
    
    Evec = [0., 0.25, 0.5, 0.75, 1]
    t0_vec = [0.25, 0.5, 0.75]
    bs = 100
else:
    name_ddpm = 'ddpm_Jul02'

    E_opt = ''
    #Evec = [0., 0.25, 0.5, 0.75, 0.9]
    #Evec = [0.8, 0.85, 0.9, 0.95, 1.]
    Evec = [0.0, 0.1, 0.2, 0.3]
    t0_vec = [0.25, 0.5, 0.75]  
    bs = 100
    
    print('automatically selecting models!')
    
print('ddpm:', name_ddpm)


bs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
npath = '../nets/'
ddpm, ac, ppath = load_models_for_evaluation(npath, name_ddpm, 'ddpm', device=device)
ddpm.model.pars['theta_ref'] = 0.95

evaluate_theta_relation(ddpm, ac, ppath, E_opt = E_opt, bs = bs, Evec = Evec, t0_vec = [0.5], Nt = 13, delta_t = 3)


