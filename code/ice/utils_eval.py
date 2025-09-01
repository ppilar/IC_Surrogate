# -*- coding: utf-8 -*-
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import gc
from scipy import stats
from pathlib import Path
import os
try:
    from .utils_generate_data import *
except:
    print('NuRadioMC not installed!')


from ._mod_conditional_denoising_diffusion_pytorch import cUnet, cGaussianDiffusion
from ._mod_conditional_denoising_diffusion_pytorch_1d import cDataset1D, Trainer1D, cUnet1D, cGaussianDiffusion1D
from .utils import *
from .utils_ddpm import *
from .utils_ac import *
from .plots import *
from .utils_vary_theta import *
from .utils_tnet import *
from .utils_load import *

def get_Etvec(E_opt = 'lowE'):
    if E_opt == 'lowE':
        Evec = [0.025, 0.25]
    else:
        Evec = [0.025, 0.25, 0.5, 0.75, 0.975]   
    tvec = [0.025, 0.25, 0.5, 0.75, 0.975]
    return Evec, tvec


def get_Et0vec(E_opt, Evec = '', t0_vec = ''):
    if Evec == '':
        if E_opt == 'lowE':
            Evec = [0., 0.25]
        else:
            Evec = [0., 0.25, 0.5, 0.75, 1]            
    if t0_vec == '':
        t0_vec = [0.25, 0.5, 0.75]  
        
    return Evec, t0_vec



def get_ddpm_name(ddpm_path, ddpm_name):
    if ddpm_name == '':
        files = [f for f in os.listdir(ddpm_path) if f.endswith('.pt')]
        ddpm_name = files[0]
        pfolder_name = get_folder_name(check_fpath(ddpm_path))        
    else:
        pfolder_name = ddpm_name
        
    if not os.path.exists(ddpm_path):
        raise Exception("Invalid path: " + ddpm_path)
    return ddpm_name, pfolder_name



def get_ppath(ddpm_path, ddpm_name):
    ddpm_path = check_fpath(ddpm_path)
    ddpm_name, pfolder_name = get_ddpm_name(ddpm_path, ddpm_name)
    if ddpm_name == 'final':
        ppath = ddpm_path + 'plots/'
    else:
        ppath = '../plots/temp_' + pfolder_name + '/'
    Path(ppath).mkdir(parents=True, exist_ok=True)
    return ppath, ddpm_name





def evaluate_ddpm(ddpm_path, ddpm_name = '', eval_opt = 'dist', E_opt = 'lowE', device='cpu'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    Evec, tvec = get_Etvec(E_opt)
    bs = 128 if eval_opt == 'samples' else 256
    tol = 1e-7 if eval_opt == 'samples' else 2.5e-2
    
    
    normalize = False
    x0, yn, a0 = load_corresponding_data(eval_opt, E_opt, lx = 1024, normalize = normalize)
    ddpm_name, pfolder_name = get_ddpm_name(ddpm_path, ddpm_name) 
    ppath, _ = get_ppath(ddpm_path, ddpm_name) 
    _, diffusion, _ = load_ddpm(ddpm_path, ddpm_name, device)
    ac = load_standard_ac(x0.shape[1], device)
    
    diffusion.model.eval()
    diffusion.is_ddim_sampling = True
    diffusion.sampling_timesteps = 50
    
    
    #####
    #####
    
    metric_names = ['minmax', 'amplitude', 'Efluence']
    metric_labels = ['log10(max-min)', 'log10(|a|)', r'log10($\psi$)']


    Np = max(len(Evec), len(tvec))
    fig2, axs2 = plt.subplots(Np, Np, figsize=(3*Np,3*Np))
    fig3, axs3 = plt.subplots(Np, Np, figsize=(3*Np,3*Np))
    fig4, axs4 = plt.subplots(Np, Np, figsize=(3*Np,3*Np))
    fig5, axs5 = plt.subplots(Np, Np, figsize=(3*Np,3*Np))
    fig6, axs6 = plt.subplots(Np**2, 2, figsize=(16,8*Np**2))
    axs = [axs2, axs3, axs4, axs5]
    figs = [fig2, fig3, fig4, fig5, fig6]
    ks_ges = []
    w1_ges = []

    for i, E in enumerate(Evec):
        for j, t in enumerate(tvec):
            print('i,j:',i,j)
            with torch.no_grad():
                tgen = time.time()
                xEt_gen, yEt_gen = get_xEt_gen(diffusion, ac, E, t, bs, tol = tol, normalize=normalize)
                print('t_per_sample=%.3f'%((time.time()-tgen)/bs))
                xEt, yEt, aEt = get_Et_samples(torch.tensor(x0), yn, a0, E, t, tol = 5e-4)
                xEt_gen = xEt_gen.cpu()
                
            print(xEt.shape)
            print(xEt_gen.shape)
            axplot_sample_comparison(axs2[i,j], xEt, xEt_gen.to('cpu'), E, t, opt = 'contour')        
            metrics = extract_metrics(xEt, metric_names)
            metrics_gen = extract_metrics(xEt_gen, metric_names)        
            print('Nr=',xEt.shape[0],'Ng=',xEt_gen.shape[0])
            for jm, (metric, metric_gen) in enumerate(zip(metrics, metrics_gen)):
                axplot_metric(axs[jm+1][i,j], metric, metric_gen, E, t, metric_labels[jm], ks_ges, w1_ges)
            axplot_real_gen_samples(axs6[Np*i + j, 0], axs6[Np*i + j, 1], xEt, xEt_gen, E, t)
            
            
        
    fnames = ['Et_contours'] + metric_names + ['samples']
    for jf, fig in enumerate(figs):
        if fnames[jf] != 'samples':
            fig.subplots_adjust(hspace=0.6, wspace=0.3)
        fig.savefig(ppath + 'tol' + str(tol) + '_' + fnames[jf] + '.pdf', bbox_inches='tight')    
    plt.show()


def print_theta_table(fpath, Et_vals, errors, errors_norm, E, theta_0, delta_t, it0, bs):
    with open(fpath, 'a') as f:        
        f.write('E='+str(reverse_E_normalization(E)))
        f.write(', theta_0='+str(theta_0))
        f.write(', delta_t='+str(delta_t))
        f.write('\n\n')
        
        formatter={'float_kind':lambda x: "%.2f" % x}        
        tvec = reverse_t_normalization(Et_vals[::bs,1]).cpu().numpy()
        
        
        captions = [r'$\theta$', r'$\mu$', r'$\sigma$', r'$\mu_n$ ', r'$\sigma_n$']
        arrays = [tvec, errors.mean(1), errors.std(1), errors_norm.mean(1), errors_norm.std(1)]
        
        for j, (cap, arr) in enumerate(zip(captions, arrays)):                
            if j > 0 and j < 3:
                fac = 1000
                fbuf = arr[it0]
                while fbuf/(fac) < 1:
                    fac /= 10
            else:
                fac = 1
                
            buf = np.array2string(arr/fac, formatter=formatter)
            if j == 0:
                f.write(r'\multicolumn{13}{c}{$\log_{10}{E}=%.2f$, $\theta_0=%.2f$°}'%(reverse_E_normalization(E), theta_0))
                f.write('\\')
                buf = buf.replace('\n', '').replace(' ', ' & ') .replace('[','').replace(']','')
                f.write(cap + ' & ' + buf + ' & ' + r'[°] \\')
                f.write(r'\cmidrule{2-12}')
            else:
                buf = buf.replace('\n', '').replace(' ', ' & ') .replace('[','').replace(']','')
                f.write(cap + ' & ' + buf + ' & (' + ('%.2e'%(fac))[-4:] + ') \\')
                
                
        f.write('\n')
        
    
def get_folder_name(path):
    i1 = 0
    found1 = False
    while found1 == False:
        i1 -= 1
        if path[i1] == '/':
            found1 = True
    
    i0 = i1
    found0 = False
    while found0 == False:
        i0 -= 1
        if path[i0-1] == '/':
            found0 = True
        
            
    return path[i0:i1]
    
    
def evaluate_theta_relation(model, ac, ppath, E_opt = 'lowE', bs = 100, Evec = '', t0_vec = '', Nt = 11, delta_t = 1, model_opt='ddpm', plot_comment='', ddpm_data=-1, device='cpu'):
    device = model.device
    Evec, t0_vec = get_Et0vec(E_opt, Evec, t0_vec)    
    
    if not 'seq_length' in ac.pars.keys():
        ac.pars['seq_length'] = 896
        
    x0, yn, a0 = load_corresponding_data('theta_rel', E_opt, ac.pars['seq_length'])
    if model_opt == 'tnet':
        yn_ges = yn
    
    #%%
    ############
    ############
    
    pfolder_name = get_folder_name(ppath)
    fname =  pfolder_name + '_bs' + str(bs) + '_Nt' + str(Nt) + '_dt'+str(delta_t)
    fpath = ppath + fname + '.txt'
    with open(fpath, 'w') as f:
        f.write('ddpm: ' + pfolder_name + '\n')
        f.write('ac: ' + ac.ac_name + '\n')
        f.write('bs: '+str(bs) + '\n')
        f.write('Nt: '+str(Nt) + '\n')
        f.write('\n\n')
            
    
    for E in Evec:
        for t0 in t0_vec:
            theta_0 = reverse_t_normalization(t0)
            print('E=', E)
            print('t0=', theta_0)             
    
            print('dt:', delta_t)
            Et_vals = get_vary_theta_pars(bs = bs, Nt = Nt, E=E, theta_0 = theta_0, delta_t = delta_t, device=device)
            if model_opt == 'ddpm':
                with torch.no_grad():
                    Et_samples_gen = generate_vary_theta_samples(model, ac, Et_vals, Nt)
            elif model_opt == 'tnet':
                model.model.to('cpu')
                model.model.eval()
                ac.model.to('cpu')
                ac.model.eval()
                
                if type(ddpm_data) == int:
                    xn, _ = normalize_x(x0)
                    xrefs, yrefs = get_ref_signals(model.pars['theta_ref'], xn, yn_ges) 
                else:
                    xrefs, yrefs = get_ref_signals(model.pars['theta_ref'], ddpm_data[0], ddpm_data[1]) 
                buf = torch.linspace(0,bs-1,bs).repeat(Nt)
                buf_Et_vals = torch.cat((Et_vals.cpu(), buf.unsqueeze(1)),1).unsqueeze(1)
                
                irefs = get_irefs(buf_Et_vals, yrefs)
                xref = xrefs[irefs]
                yref = yrefs[irefs]
                with torch.no_grad():
                    Et_samples_gen = generate_vary_theta_samples_TNet(model, xref, yref, ac, buf_Et_vals, Nt, device = 'cpu').detach()
                   
            
            
            #%%        
            it0 = Nt-1
            Et_vals2, iN, ibuf = get_closest_match(Et_samples_gen, x0, yn, Et_vals, it0, bs)
            plot_closest_matches(Et_samples_gen.cpu(), Et_vals.cpu(), x0, ibuf, it0, bs)
            
            
            #%% data generation code
            parameters_scaled = get_NuRadioMC_parameters(Et_vals2.cpu().numpy())
            t = time.time()
            iN_vals = iN[...,np.newaxis].repeat(Nt,1).T.reshape(-1)
            if not 'dt' in ac.pars.keys():
                if ac.pars['seq_length'] == 1024:
                    ac.pars['dt'] = 5e-10
                else:
                    ac.pars['dt'] = 1e-10
            signals_filtered, _, Nsp, _ = generate_signals(parameters_scaled, iN_vals = iN_vals, N_timebin = ac.pars['seq_length'], dt = ac.pars['dt'])
            print('t:', time.time() - t)
            #%%            
            ip = 10
            plot_vary_theta_comparison(Et_samples_gen.cpu(), signals_filtered[:,3:], parameters_scaled, bs, Nsp, iN[:ip], ppath=ppath, normalize_plot=0, plot_comment=plot_comment)
            plot_vary_theta_comparison(Et_samples_gen.cpu(), signals_filtered[:,3:], parameters_scaled, bs, Nsp, iN[:ip], ppath=ppath, normalize_plot=1, plot_comment=plot_comment)
                
            #%%
            gen_max = Et_samples_gen.abs().max(1)[0].cpu().numpy()
            signals_max = np.abs(signals_filtered[:,3:]).max(1)
            errors, errors_norm = np.zeros((2, Nt, bs))
            for j in range(bs):
                for k in range(Nt):
                    errors[k,j] = np.abs(Et_samples_gen[k*bs + j, :].cpu().numpy() - signals_filtered[bs*k + Nsp*j,3:]).sum()
                    errors_norm[k,j] = np.abs(Et_samples_gen[k*bs + j, :].cpu().numpy()/gen_max[k*bs + j] - signals_filtered[bs*k + Nsp*j,3:]/signals_max[bs*k + j]).sum()
                    
            
            print(errors.mean(1))
            print(errors_norm.mean(1))
            print(errors_norm.std(1))
            
            #%%
            print_theta_table(fpath, Et_vals, errors, errors_norm, E, theta_0, delta_t, it0, bs)
                         
            plt.close('all')
            gc.collect()  # Force garbage collection
    
    
    
    

def evaluate_theta_relation_detailed(tnet, data, ppath=''):
    xn, yn = data
    
    tvec_axis, tlabel = get_tvec(xn.shape[-1])    
    Evec = [0., 0.1, 0.25]
    tvec = [0.05, 0.25, 0.5, 0.75, 0.95]
    Np = max(len(tvec),len(Evec))
    fig1, axs1 = plt.subplots(Np, Np, figsize=(3*Np, 3*Np))
    fig2, axs2 = plt.subplots(Np, Np, figsize=(3*Np, 3*Np))
    
    tol_E = 5e-2#2.5e-2
    tol_t = 0.005
    for i, E in enumerate(Evec):
        for j, t in enumerate(tvec):
            inds = get_ref_inds(tnet.pars['theta_ref'], yn)
            xrefs, yrefs = get_ref_signals(tnet.pars['theta_ref'], xn, yn)
            iEbuf = np.where(np.abs(yrefs[:,0] - E) < tol_E)[0]
            xrefs_E = xrefs[iEbuf]
            yrefs_E = yrefs[iEbuf]
            it = np.where(np.abs(yn[:161,1] - t) < tol_t)[0].item()
            it_ref = np.where(np.abs(yn[:161,1] - tnet.pars['theta_ref']) < tol_t)[0].item()
            dt = it_ref - it
            
            xrefs_E = torch.tensor(xrefs_E).float()
            yrefs_E = torch.tensor(yrefs_E).float()
            
            xpred, _ = tnet.model((xrefs_E, yrefs_E), yrefs_E[:,0], torch.tensor(t*np.ones(xrefs_E.shape[0])).float())
            xpred = xpred.detach().numpy()
            xtrue = xn[inds[iEbuf] - dt]
            
            abs_error = np.abs(xpred - xtrue)
            rel_error = np.abs((xpred - xtrue)/(xtrue+1e-12))
            
            axs1[i,j].hist(abs_error.sum(1), range = (0, 2), density=True)
            axs1[i,j].set_title(get_Et_title(E,t))
            axs1[i,j].set_xlabel(r'$\int |x - \hat x| dt$')
            
            
            axs2[i,j].plot(tvec_axis, abs_error.mean(0), label=r'$|x - \hat x|$')
            axs2[i,j].set_title(get_Et_title(E,t))
            axs2[i,j].set_ylim([0,0.05])
            axs2[i,j].set_xlabel(tlabel)
            if i == 0 and j == 0:
                axs2[i,j].legend()
            
    fig1.subplots_adjust(hspace=0.6, wspace=0.3)
    fig2.subplots_adjust(hspace=0.6, wspace=0.3)

    fig1.savefig(ppath + 'abs_error_hist.pdf', bbox_inches='tight')
    fig2.savefig(ppath + 'abs_error_vs_t.pdf', bbox_inches='tight')
    

    