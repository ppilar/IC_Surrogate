# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path
from scipy import stats

from ice.utils import *
from ice.utils_ac import *
from ice.plots import axplot_value_dist, axplot_error_dist, axplot_amplitude_dist
from ice.utils_load import *

#%%
if len(sys.argv) > 1:
    name_ac = sys.argv[1]
else:
    name_ac = 'ac_May30'
    print('automatically selecting models!')    

print('ac:', name_ac)

ppath = '../plots/temp_' + name_ac + '/'
Path(ppath).mkdir(parents=True, exist_ok=True)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device='cpu'

#%% load data + models
print('load model + data')
ac = load_ac('../nets/', name_ac, device)
xn, yn, a0, _ = load_data(get_data_path(),  shorten_seq=False, remove_invalid = True, normalize = True)
data = [torch.tensor(z).to(device) for z in [xn, np.log10(a0), yn]]

#%% evaluate
#evaluate_ac(ac, data, ppath)
inputs_train, outputs_train, inputs_val, outputs_val = ac.train_validation_split(data)
with torch.no_grad():
    ahat_train = ac.forward(inputs_train.detach())[0].cpu().numpy() 
    ahat_val = ac.forward(inputs_val.detach())[0].cpu().numpy() 

#%%   
delta_log10_a_train = ahat_train - outputs_train.numpy()
delta_log10_a_val = ahat_val - outputs_val.numpy()

bounds = (-0.1, 0.1)
fig, axs = plt.subplots(1,2,figsize=(9,3))
axplot_amplitude_dist(axs[0], (outputs_train.numpy(), outputs_val.numpy()))
axplot_error_dist(axs[1], (delta_log10_a_train, delta_log10_a_val))
axs[0].set_ylabel('normalized counts')

plt.savefig(ppath+'error_dist.pdf', bbox_inches='tight')

#%%
def get_metrics(buf):
    if type(buf) == torch.Tensor:
        buf = buf.numpy()
    return (buf.mean(), buf.std(), np.quantile(buf, 0.05), np.quantile(buf, 0.95))

def get_eval_str(buf):
    return '%.3f \pm %.3f; [%.3f, %.3f]' % buf

print('dl10_train:  ' + get_eval_str(get_metrics(delta_log10_a_train)))
print('dl10_test: ' + get_eval_str(get_metrics(delta_log10_a_val)))
print('|dl10_train|: ' + get_eval_str(get_metrics(np.abs(delta_log10_a_train))))
print('|dl10_test|: ' + get_eval_str(get_metrics(np.abs(delta_log10_a_val))))
