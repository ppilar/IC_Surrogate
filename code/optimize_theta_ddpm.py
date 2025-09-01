#-*- coding: utf-8 -*-
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import gc
import tqdm

from ice._mod_conditional_denoising_diffusion_pytorch import cUnet, cGaussianDiffusion
from ice._mod_conditional_denoising_diffusion_pytorch_1d import cDataset1D, Trainer1D, cUnet1D, cGaussianDiffusion1D
from ice.utils import *
from ice.utils_ddpm import *
from ice.utils_ac import *
from ice.plots import *
from ice.utils_vary_theta import *
from ice.utils_eval import *
from ice.utils_tnet import *

#%%
def area(sgen):
    return (sgen-0.5).abs().sum(-1)

#%%
init_random_seeds(s=42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
Evec, tvec = get_Etvec('low_E')

#%%
pars = init_ds_pars()
#pars = init_ds_pars('theta_rel_low_E')
#xn, yn, _ = load_data(pars['path_filtered'], Ns = pars['Ns'], shorten_seq=False, remove_invalid = pars['remove_invalid'], normalize = True)
xn, yn, _, _ = load_data(pars['path_filtered'], shorten_seq=False, remove_invalid = pars['remove_invalid'], normalize = True)


#%% load model
model_opt = 'ddpm'
if model_opt == 'ddpm':
    npath = '../nets/'
    #ddpm_name = 'ddpm_size_3'
    ddpm_name = 'ddpm_Jul02'
    ddpm, _, _ = load_models_for_evaluation(npath, ddpm_name, model_opt, device=device)
if model_opt == 'tnet':
    #model_path = '../runs/runs_theta_transformation/ref_Sep12_22-55-24_Block_v3'
    #tnet, ac, ppath = load_models_for_evaluation(model_path, '', model_opt, device=device)
    
    npath = '../nets/'
    tnet_name = 'tnet_Jun19'
    #tnet_name = 'tnet_Feb16'
    tnet, ac, ppath = load_models_for_evaluation(npath, tnet_name, model_opt, device=device)
    xrefs, yrefs = get_ref_signals(tnet.pars['theta_ref'], xn, yn)
    

#%% create classes
loss_ges_ges = []
theta_ges_ges = []
#energies = [0.5, 0.5]#[0.0, 0.0, 0.3, 0.3]
#energies = [0.5, 0.5, 0.8, 0.8]
#start_thetas = [0.1, 0.9]


#NE = 1
#energies = [0.]*2
#start_thetas = [0.1] + [0.9]#[0.3, 0.7, 0.3, 0.7]

NE = 4
energies = [0.0, 0.25, 0.5, 0.75]*2
start_thetas = [0.1]*NE + [0.9]*NE#[0.3, 0.7, 0.3, 0.7]

#itges = 200 if model_opt == 'tnet' else 200
itges = 800
for energy, start_theta in zip(energies, start_thetas):

    theta = torch.tensor(start_theta, requires_grad = True, device = device)
    mask = torch.tensor([[0,1]], device = device)    
    classes0 = torch.tensor([[energy, 0.]], device = device).float()
    bs = 10 #32 #TODO: enable bs>10 for tnet
    classes0 = classes0.repeat(bs,1)
    
    
    #%% optimization loop
    loss_ges = []
    theta_ges = []
    optimizer = optim.Adam([theta], lr=0.01, betas=(0.95,0.99))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    
    #pbar = tqdm(["a", "b", "c", "d"])
    #for char in pbar:
    #    pbar.set_description("Processing %s" % char)
    tavg = 0
    pbar = tqdm(range(itges))
    for j in pbar:
        tj = time.time()
        optimizer.zero_grad()
        classes = classes0 + theta*mask
        
        if j%150 == 0 and j > 0:
            scheduler.step()
        
        #generate samples
        if model_opt == 'ddpm':
            sgen, _ = sample_from_ddpm(ddpm, bs, classes=classes)
        if model_opt == 'tnet':
            Et = classes.to(device) #TODO: rename classes -> Et
            buf = torch.linspace(0,bs-1,bs, device=device)#.repeat(Nt)
            Et = torch.cat((Et, buf.unsqueeze(1)),1)
            
            xref, yref = get_xy_ref(Et.detach().cpu(), xrefs, yrefs)
            xref = torch.tensor(xref, device=device).float()
            yref = torch.tensor(yref, device=device).float()
            
            #xbase = transform_signal(xref, yref[:,1], Et[:,1])
            #sgen = tnet.model(xbase.unsqueeze(1), Et[:,0], Et[:,1]).squeeze()
            xbase = (xref, yref)
            sgen = tnet.model(xbase, Et[:,0], Et[:,1])[0].squeeze()
        
        # if j%50 == 0:            
        #     plt.figure()
        #     plt.plot(sgen.detach().cpu()[:,450:550].T)
        #     plt.show()
            
        use_a = False
        if use_a == True:
            a = ac.forward(torch.cat((sgen, yref), 1).unsqueeze(1))[0]
            sgen = sgen*10**a.unsqueeze(1)
        
        if use_a == True:
            loss = -sgen.abs().max(1)[0].mean()
        else:
            loss = area(sgen).mean()
        loss.backward()
        optimizer.step()
        #print('j:', j, 'l:', loss.item(), 't:', theta.item())
        loss_ges.append(loss.item())
        theta_ges.append(reverse_t_normalization(theta.item()))
        
        tavg = (tavg*j + (time.time()-tj))/(j+1)
        pbar.set_description('j:%i l:%.3f t:%.3f tavg:%.3f'%(j, loss.item(), reverse_t_normalization(theta.item()), tavg))
    
    loss_ges_ges.append(loss_ges)
    theta_ges_ges.append(theta_ges)
    print('theta:', reverse_t_normalization(theta.detach().cpu().numpy()))
    print('tavg:', tavg)

#%%
with open('../results/optim_'+model_opt+'.pk', 'wb') as f:
    pickle.dump([loss_ges_ges, theta_ges_ges, energies, start_thetas, NE], f)
    
# with open('../results/optim_ddpm_v1.pk', 'rb') as f:
#     loss_ges_ges, theta_ges_ges, energies, start_thetas, NE = pickle.load(f)

#%% plot theta trajectory
plt.rcParams['font.size'] = 14  # Set global font size
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Helvetica']  # Optional: specific font
plt.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text





fig, axs = plt.subplots(1,2, figsize = (10,4))
for loss_ges in loss_ges_ges:
    axs[0].plot(loss_ges, label='loss')
axs[0].set_title('loss')
axs[0].set_xlabel('iteration')
for i, theta_ges in enumerate(theta_ges_ges):
    #axs[1].plot(theta_ges, label=r'$\theta$=%.2f°'%(theta_ges_ges[i][-1]))
    label = r'E=%.2eeV'%(10**reverse_E_normalization(energies[i])) if i < NE else ''
    axs[1].plot(theta_ges, label=label, color='C'+str(i%NE))
    #axs[1].plot(theta_ges, label=r'E=%.2eeV, $\theta_0$=%.2f°'%(10**reverse_E_normalization(energies[i]),reverse_t_normalization(start_thetas[i])))
#axs[1].set_title(r'$\theta$')
axs[1].hlines(55.82,0,itges,color='grey',linestyle='--')
axs[1].legend()
axs[1].set_xlabel('iteration')
axs[1].set_ylabel(r'$\theta$ [°]')
axs[1].set_ylim([35,75])
if model_opt == 'tnet':
    axs[1].set_title(r'$\theta$-Net')
elif model_opt == 'ddpm':
    axs[1].set_title(r'DDPM')
fig.savefig('../plots/optimize_theta_test/loss-' + model_opt + '.pdf', bbox_inches = 'tight')



#%%
# with torch.no_grad():
#     Et2 = Et.repeat(10,1).detach()
#     Et2[:,-1] = 0
#     Et2[:,1] = torch.linspace(0.4,0.6, Et2.shape[0])
    
#     xref, yref = get_xy_ref(Et2.detach().cpu(), xrefs, yrefs)
#     xref = torch.tensor(xref, device=device).float()
#     yref = torch.tensor(yref, device=device).float()
    
#     #xbase = transform_signal(xref, yref[:,1], Et[:,1])
#     #sgen = tnet.model(xbase.unsqueeze(1), Et[:,0], Et[:,1]).squeeze()
#     xbase = (xref, yref)
#     sgen = tnet.model(xbase, Et2[:,0], Et2[:,1])[0].squeeze()
    
#     plt.figure(figsize=(10,8))
#     plt.plot(sgen.cpu().T[490:530])
#     plt.figure(figsize=(10,8))
#     plt.plot(xn[70:90, 490:530].T)








