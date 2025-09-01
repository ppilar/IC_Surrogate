# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from ice.utils import load_data
from ice.utils_tnet import *
from ice.utils_ddpm import init_ds_pars
from ice.utils_eval import evaluate_theta_relation, load_models_for_evaluation, evaluate_theta_relation_detailed
from ice.utils_load import get_standard_tnet_path
from ice.plots import axplot_hist_ncounts


class ResTNet(object):
    def __init__(self, yn, inds_train, inds_val):
        self.yn = yn
        self.inds_train = inds_train
        self.inds_val = inds_val


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%% load data
pars = init_ds_pars()
xn, yn, a0, _ = load_data(pars['path_filtered'], shorten_seq=False, remove_invalid = pars['remove_invalid'], normalize = True)

#%% load net
model_path = '../runs/runs_theta_transformation/ref_Jun19_00-43-42_UNet'
#model_path = '../runs/runs_theta_transformation/_Aug02_00-01-46_UNet' #extract_features = False
#model_path = '../runs/runs_theta_transformation/_Aug02_00-02-28_UNet' #transform_signals = False


if not 'model_path' in locals():
    model_path = get_standard_tnet_path(pars['ds_opt'])     
tnet, _, ppath = load_models_for_evaluation(model_path, 'final', 'tnet', seq_length=xn.shape[1], device='cpu')

#ppath = model_path + '/plots/'

#%% calculate evaluation metrics

inds_val, inds_train = get_bool_array(yn.shape[0], tnet.ival)
iref = get_ref_inds(tnet.pars['theta_ref'], yn)
xrefs, yrefs = get_ref_signals(tnet.pars['theta_ref'], xn, yn)
tnet.model.to(device)
xrefs = torch.tensor(xrefs).to(device)
yrefs = torch.tensor(yrefs).to(device)
yn = torch.tensor(yn).to(device)


dt = 0.5 #[ns] ... spacing of time grid
freqs = np.fft.rfftfreq(xrefs.shape[1], d=dt) #[GHz]
di = 160 #number of samples before next reference signal
ndi = 4 #number of subbatches
dx_abs, df_abs, dt_max, df_max = np.zeros((4, yn.shape[0]))
with torch.no_grad():
    for j in tqdm(range(410*ndi)):
        x_pred, x_base = tnet.model((xrefs[j//ndi].unsqueeze(0).repeat(di//ndi,1), yrefs[j//ndi].unsqueeze(0).repeat(di//ndi,1)), yn[j*di//ndi:(j+1)*di//ndi, 0], yn[j*di//ndi:(j+1)*di//ndi, 1])
        x_pred = x_pred.cpu().numpy()
        x_base = x_base.cpu().numpy()
        x_true = xn[j*di//ndi:(j+1)*di//ndi,:]
    
        dx_abs[j*di//ndi:(j+1)*di//ndi] = np.abs((x_pred - x_true)).sum(1)/np.abs(x_true).sum(1)
        df_abs[j*di//ndi:(j+1)*di//ndi] = np.abs(np.abs(np.fft.rfft(x_pred, axis=1)) - np.abs(np.fft.rfft(x_true, axis=1))).sum(1)/np.abs(np.fft.rfft(x_true, axis=1)).sum(1)      
        dt_max[j*di//ndi:(j+1)*di//ndi] = (np.abs(x_pred).argmax(1) - np.abs(x_true).argmax(1))*dt
        df_max[j*di//ndi:(j+1)*di//ndi] = freqs[np.abs(np.fft.rfft(x_pred, axis=1))[:,1:].argmax(1)] - freqs[np.abs(np.fft.rfft(x_true, axis=1))[:,1:].argmax(1)]
        
#%% 
        
res_tnet = ResTNet(yn, inds_train, inds_val)
res_tnet.dx_abs = dx_abs
res_tnet.df_abs = df_abs
res_tnet.dt_max = dt_max
res_tnet.df_max = df_max

with open('../results/res_tnet.pk', 'wb') as f:
    pickle.dump(res_tnet, f)
    
    
#%% 
titles = [r'$\Delta x_0$', r'$\Delta \tilde x_0$', r'$\Delta  \, \max  \, x_0 \, [\rm ns]$', r'$\Delta  \, \max \, \tilde x_0 \, [\rm GHz]$']
xlims = [[0,0.02], [0,0.2], [-5,5], [-0.05, 0.05]]
fig, axs = plt.subplots(1,4, figsize=(15,3))
for inds in [res_tnet.inds_train.to(bool), res_tnet.inds_val.to(bool)]:    
    axplot_hist_ncounts(axs[0], res_tnet.dx_abs[inds], 51, xlims=xlims[0], alpha=0.7)
    axplot_hist_ncounts(axs[1], res_tnet.df_abs[inds], 51, xlims=xlims[1], alpha=0.7)
    axplot_hist_ncounts(axs[2], res_tnet.dt_max[inds], 51, xlims=xlims[2], alpha=0.7)
    axplot_hist_ncounts(axs[3], res_tnet.df_max[inds], 51, xlims=xlims[3], alpha=0.7)
    
axs[0].legend(['train', 'test'])
axs[0].set_ylabel('normalized counts')
for j, ax in enumerate(axs):
    ax.set_xlim(xlims[j])
    ax.set_xlabel(titles[j])

#ppath = '../plots/t-Net/'
plt.savefig(ppath + 'eval_dist.pdf', bbox_inches='tight')

#%%
def get_metrics(buf):
    if type(buf) == torch.Tensor:
        buf = buf.numpy()
    return (buf.mean(), buf.std(), np.quantile(buf, 0.05), np.quantile(buf, 0.95))

def get_eval_str(buf, opt=''):
    if opt == '':
        return '%.3f \pm %.3f; [%.3f, %.3f]' % buf
    elif opt == 'std':
        return '%.3f \pm %.3f' % buf[:2]  
    elif opt == 'quantiles abs':
        return '%.3f' % buf[3]    
    elif opt == 'quantiles':
        return '[%.3f, %.3f]' % buf[2:]

with open(ppath + 'eval.txt', 'w') as f:
    for i, inds in enumerate([res_tnet.inds_train.to(bool), res_tnet.inds_val.to(bool)]):
        buf = 'train' if i == 0 else 'test'
        f.write(buf)
        f.write('\ndx_abs:  ' + get_eval_str(get_metrics(res_tnet.dx_abs[inds])))
        f.write('\ndf_abs:  ' + get_eval_str(get_metrics(res_tnet.df_abs[inds])))
        f.write('\ndx_max:  ' + get_eval_str(get_metrics(res_tnet.dt_max[inds])))
        f.write('\ndf_max:  ' + get_eval_str(get_metrics(res_tnet.df_max[inds])))
        f.write('\n\n')
        f.write(buf + ' & ' + get_eval_str(get_metrics(res_tnet.dx_abs[inds]), 'quantiles abs')
                    + ' & ' + get_eval_str(get_metrics(res_tnet.df_abs[inds]), 'quantiles abs')
                    + ' & ' + get_eval_str(get_metrics(res_tnet.dt_max[inds]), 'quantiles')
                    + ' & ' + get_eval_str(get_metrics(res_tnet.df_max[inds]), 'quantiles'))
        f.write('\n\n')
