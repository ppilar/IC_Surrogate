# -*- coding: utf-8 -*-
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from ice.utils_ac import *
from ice.utils import init_par, print_parameters
from ice.utils import *
from ice.plots import *


if not 'pars' in locals(): pars = dict()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

#%% load data and initialize parameters
pars['ds_opt'] = 'EM_grid'
pars['path_filtered'] = get_data_path(lx=1024)#get_path_filtered(ds_opt=pars['ds_opt'])
xn, yn, a0, data_properties = load_data(pars['path_filtered'], shorten_seq=False, remove_invalid = True, normalize = True)
init_standard_pars_ac(pars, data_properties)


#%% initialize or load AC
load_ac = 0
if load_ac == 1:
    AC = AmplitudeClassifier()
    AC.load('../nets/' + 'ac_test.pt')
else:
    AC = AmplitudeClassifier(pars)

#%% train AC
data = [torch.tensor(z).float() for z in [xn, np.log10(a0), yn]]
AC.train(data, device=device)


#%% evaluate AC
AC.model.eval()
AC.model.to('cpu')
data[1] = torch.tensor(np.log10(a0)).float()
evaluate_ac(AC, data, AC.writer.log_dir)