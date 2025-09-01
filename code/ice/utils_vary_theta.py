# -*- coding: utf-8 -*-
import torch
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pickle
try:
    from utils_generate_data import *    
except:
    print('NuRadioMC not installed!')

from ._mod_conditional_denoising_diffusion_pytorch import cUnet, cGaussianDiffusion
from ._mod_conditional_denoising_diffusion_pytorch_1d import cDataset1D, Trainer1D, cUnet1D, cGaussianDiffusion1D
from .utils import *
from .utils_ddpm import *
from .utils_ac import *
from .plots import *

plt.rcParams['font.size'] = 14  # Set global font size
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Helvetica']  # Optional: specific font
plt.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text


def get_vary_theta_pars(bs = 10, Nt = 11, E = 0.5, theta_0 = 45.82, delta_t = 1, device = 'cpu'):
    Evals = torch.tensor(E*np.ones(Nt*bs)).unsqueeze(1)

    if Nt%2== 1:
        tvals = theta_0 + torch.linspace(-delta_t*int((Nt-1)/2),delta_t*int((Nt-1)/2),Nt).repeat(bs,1).T.reshape(-1).unsqueeze(1)
    else:
        if delta_t == 1:
            tvals = theta_0 + torch.linspace(-int((Nt-2)/2),int((Nt)/2),Nt).repeat(bs,1).T.reshape(-1).unsqueeze(1)
        else:
            tvals = theta_0 + torch.linspace(-delta_t*int((Nt-2)/2),delta_t*int((Nt)/2),Nt).repeat(bs,1).T.reshape(-1).unsqueeze(1)
    tvals = normalize_t(tvals)
    Et_vals = torch.cat((Evals, tvals),1).float().to(device)
    return Et_vals

def get_vary_E_pars(bs = 10, Ni = 11, E0 = 0.9, dE = 0.02, theta=0.9, device='cpu'):
    Evals = torch.linspace(E0,E0+dE,Ni).repeat(bs,1).T.reshape(-1).unsqueeze(1)
    tvals = theta*torch.ones(Ni*bs).unsqueeze(1)
    Et_vals = torch.cat((Evals, tvals), 1).float().to(device)
    return Et_vals

def get_vary_epsilon_pars(seq_length, bs = 10, Ni = 11, E = 0.9, theta = 0.9, device='cpu'):
    epsilon0 = torch.randn(bs,seq_length)
    epsilon1 = torch.randn(bs,seq_length)
    fac = torch.linspace(0,1,Ni).unsqueeze(1).repeat(1,bs).reshape(-1,1)
    epsilon = (epsilon0.repeat(Ni,1)*(1-fac) + fac*epsilon1.repeat(Ni,1)).unsqueeze(1).to(device)
    E = torch.tensor([E])
    t = torch.tensor([theta])
    Et_vals = torch.cat((E,t)).repeat(Ni*bs,1).to(device)
    return Et_vals, epsilon

def generate_vary_theta_samples(diffusion, ac, Et_vals, Nt, epsilon = False):
    device = diffusion.device
    
    bs = int(Et_vals.shape[0]/Nt)
    if type(epsilon) is bool and Nt > 1:
        epsilon = torch.randn(bs,diffusion.seq_length).repeat(Nt,1).unsqueeze(1).to(device)
    diffusion.n_batch_signals = Nt
    with torch.no_grad():
        sbuf, _ = sample_from_ddpm(diffusion, Nt*bs, classes=Et_vals, epsilon=epsilon)
        
    log_ar, log_a1r, anr = ac.forward(torch.cat((sbuf.squeeze(), Et_vals),-1).unsqueeze(1))
    Et_samples_gen = reverse_sample_normalization(sbuf.squeeze(), log_ar).to(device)
    return Et_samples_gen

def generate_vary_theta_samples_TNet(TNet, xref, yref, ac, Et_vals, Nt, device = 'cpu'):
    from .utils_tnet import transform_signal, ThetaNet
    bs = int(Et_vals.shape[0]/Nt)
    
    xref = torch.tensor(xref).float()
    yref = torch.tensor(yref).float()
    sbuf, _ = TNet.model((xref, yref), Et_vals[:,0,0], Et_vals[:,0,1])
        
    log_ar, log_a1r, anr = ac.forward(torch.cat((sbuf.squeeze(), Et_vals[:,0,:2]),-1).unsqueeze(1))
    Et_samples_gen = reverse_sample_normalization(sbuf.cpu().squeeze(), log_ar)
    return Et_samples_gen


def get_closest_match(Et_samples_gen, x0, yn, Et_vals, it0, bs,normalize=True):
    if normalize:
        x0, _ = normalize_x(x0)
        Et_samples_gen, _ = normalize_x(Et_samples_gen)    
    Nt = int(Et_vals.shape[0]/bs)
    Et_vals2 = Et_vals
    ibuf = []
    iNs = []
    for j in range(bs):
        xbuf = Et_samples_gen[it0*bs+j]
        buf = np.abs(x0 - xbuf.detach().cpu().numpy()).sum(1)
        ibuf.append(np.argmin(buf))
        E, t, iN = yn[ibuf[-1]]
        Et_vals2[j::bs,0] = E*torch.ones(Nt)
        iNs.append(int(iN))
        
    return Et_vals2, np.array(iNs), ibuf

def get_i_closest_match(samples, xn):
    inds = []
    deltas = []
    for j in range(samples.shape[0]):
        buf = torch.abs(xn - samples[j].unsqueeze(0)).sum(1)
        inds.append(torch.argmin(buf).item())
        deltas.append(buf[inds[-1]].item())
    return np.array(inds), np.array(deltas)

def plot_closest_matches(Et_samples_gen, Et_vals, x0, ibuf, it0, bs):
    for j in range(bs):
        xbuf = Et_samples_gen[it0*bs+j]
        xbuf2 = x0[ibuf[j]]
        E, t = Et_vals[j]
        
        plt.figure()
        plt.plot(xbuf, label='ddpm')
        plt.plot(xbuf2, label = 'real')
        plt.title(get_Et_title(E,t))
        plt.legend()
        
def get_NuRadioMC_parameters(Et_vals):
    Evals = reverse_E_normalization(Et_vals[:,0])
    tvals = reverse_t_normalization((Et_vals[:,1]))
    parameters_scaled = np.zeros((Et_vals.shape[0], 2))
    parameters_scaled[:,0] = 10**Evals
    parameters_scaled[:,1] = tvals
    return parameters_scaled

def plot_vary_theta_comparison(Et_samples_gen, signals_filtered, parameters_scaled, bs, Nsp, iN, ppath = '', normalize_plot=0, plot_comment=''):
    Evals = np.log10(parameters_scaled[:,0])
    tvals = parameters_scaled[:,1]
    Nt = int(Evals.shape[0]/bs)
    
    plt.figure()
    fig, axs = plt.subplots(len(iN),1, figsize=(12,6*len(iN)))
    fig.subplots_adjust(hspace=0.3)
    for k in range(Nt):
        for j in range(len(iN)):
            i0 = 0 
            #i1 = signals_filtered.shape[1]#896#450
            i1 = Et_samples_gen.shape[1]
            #i0 = 250
            #i1 = 920
            i0 = 450
            i1 = 590
            #i0 = 300#0#400
            #i1 = 650#896#500
            
            tvec, tlabel = get_tvec(signals_filtered.shape[-1])
            if signals_filtered.shape[1] != Et_samples_gen.shape[1]:
                signals = signals_filtered[k*bs + j,i0+3:i1+3] 
            else:
                signals = signals_filtered[k*bs + j,i0:i1]
            if normalize_plot == 1:
                norm_true = np.max(np.abs(signals),0)
                norm_gen = np.max(np.abs(Et_samples_gen[k*bs + j,i0:i1].cpu().numpy()))
            else:
                norm_true = 1
                norm_gen = 1
            p = axs[j].plot(tvec[i0:i1], signals/norm_true, label=r'$\theta$=%.2f°'%(tvals[k*bs + j]), alpha=0.5)
            print(k*bs + j)
            axs[j].plot(tvec[i0:i1],Et_samples_gen[k*bs + j,i0:i1].cpu()/norm_gen, '--', color = p[-1].get_color())
            #print(Nsp*bs*k + Nsp*j + iN[j])
            
            
            if k == Nt-1:
                axs[j].legend(loc='upper right', fontsize=12)
                axs[j].set_xlabel(tlabel)
                axs[j].set_ylabel('normalized amplitude')
                if normalize_plot == 0:
                    if Evals[j] == 16:
                        axs[j].set_ylim([-1.3e-5, 1.3e-5])
                    elif Evals[j] == 17:
                        axs[j].set_ylim([-0.02, 0.025])
                    #elif Evals[j] == 18:
                    #    axs[j].set_ylim([-1.5e-4, 1.5e-4])
                    
            axs[j].set_title(r'E=%.2eeV'%(10**Evals[j]))
            
    delta_t = parameters_scaled[bs,1]-parameters_scaled[0,1]
    fig.savefig(ppath + plot_comment + '_theta_plot_norm'+str(normalize_plot)+'_logE%.2f_t%.2f_Nt%i_dt%.2f.pdf'%(Evals[0],tvals[int(len(tvals)/2)], Nt, delta_t), bbox_inches='tight')
  
        