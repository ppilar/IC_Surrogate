# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
#from .utils import log_prob_gaussian


def h_poly(t):
    tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt

# def interp_v0(x, y, xs):
#     m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
#     m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
#     idxs = torch.searchsorted(x[1:].detach(), xs.detach())
#     dx = (x[idxs + 1] - x[idxs])
#     hh = h_poly((xs - x[idxs]) / dx)
#     return hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx

def interp(x, y, xs, fill_value = 0.5): #not compatible with vmap!
    res = torch.zeros(xs.shape[0], device=x.device)

    #fill values outside of range of x with fill_value
    mask = (xs > x.min()) & (xs < x.max())
    #res[~mask] = fill_value
    #xs = xs[mask]
    res = torch.where(mask, res, fill_value) #vmap
    xs = torch.masked_select(xs, mask)    
    
    #interpolation inside of range of x
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[1:].detach(), xs.detach())
    dx = (x[idxs + 1] - x[idxs])
    hh = h_poly((xs - x[idxs]) / dx)
    res[mask] = hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx
    
    return res

def interp_vmap(y, scale=1.0):
    N = y.shape[-1]
    x = torch.linspace(-1, 1, N, device=y.device).expand_as(y)
    xq = torch.linspace(-1, 1, N, device=y.device).unsqueeze(0) * scale.unsqueeze(1)
    xq = xq.clamp(-1, 1).expand_as(y)
    
    
    N = x.shape[-1]
    M = xq.shape[-1]

    # Clamp xq to be within x's bounds
    xq_clamped = xq.clamp(min=x[..., 0:1], max=x[..., -1:])

    # Get indices of the interval to the right
    idx = torch.searchsorted(x, xq_clamped, right=True)
    idx = idx.clamp(min=1, max=N - 1)

    # Left/right x and y
    x0 = x.gather(-1, idx - 1)
    x1 = x.gather(-1, idx)
    y0 = y.gather(-1, idx - 1)
    y1 = y.gather(-1, idx)

    # Interpolation weight
    t = (xq_clamped - x0) / (x1 - x0 + 1e-12)
    yq = y0 + t * (y1 - y0)
    return yq
    

# def interp_vmap(y: torch.Tensor, scale: float): #not compatible with forward mode!
#     if y.ndim == 2:
#         y = y.unsqueeze(0)
#     B, C, L = y.shape
#     L_out = L
#     #print(scale)
    
#     # Create sampling grid in [-1, 1]
#     #grid = torch.linspace(-1, 1, L_out, device=y.device)
#     grid0 = torch.linspace(-1, 1, L, device=y.device)
#     grid = grid0*scale*2#2#*2*torch.pi/2
#     grid = grid.view(B, 1, -1, 1).repeat(1,1,1,2)  # shape (1, 1, L_out, 1)

#     # grid_sample expects (N, C, L, 1), so add batch and H dims
#     y = y.view(B, C, L, 1).repeat(1,1,1,2)   # (1, C, L, 1)
#     sampled = torch.nn.functional.grid_sample(y, grid, mode='bilinear', align_corners=True, padding_mode='zeros')


#     return sampled.view(B, C, L_out)  # Remove batch and extra dims
    


def get_f_interp(x, y):
    def f_interp(x2):
        return interp(x, y, x2)
    return f_interp

def get_MMD_gterm(pars, cval_gbatch, ptrue_rep):
    Nc = cval_gbatch.shape[1]
    bs = cval_gbatch.shape[0]
    print('Nc=',Nc)
    
    gterm = torch.zeros(Nc, device = cval_gbatch.device)
    for js, s0 in enumerate(pars['MMD_sig_vec']):
        stds = pars['MMD_sig_vec'][js]/ptrue_rep.fsig_best
        
        if pars['kernel_opt'] == 'G':
            buf = torch.exp(log_prob_gaussian(cval_gbatch.unsqueeze(1), cval_gbatch.unsqueeze(0), stds.unsqueeze(0).unsqueeze(0)))
        else:
            print('kernel not implemented!')
        
        gterm = gterm + buf.sum(0).sum(0) - buf.diagonal(dim1=0, dim2=1).sum(1)
        
    gterm = gterm*0.5/(bs*(bs-1))/len(pars['MMD_sig_vec'])
    return gterm

def get_MMD_gdterm(pars, cval_gbatch, ptrue_rep):
    Nc = cval_gbatch.shape[1]
    bs = cval_gbatch.shape[0]
    
    gdterm = torch.zeros(Nc, device = cval_gbatch.device)
    for jc in range(Nc):
        buf = ptrue_rep.f_interp_vec[jc](cval_gbatch[:,jc])
        gdterm = gdterm + buf.sum(0).sum(0)    
    gdterm = - 2/bs**2*gdterm
    return gdterm

def get_MMD_gdterm_v2(pars, ds, cval_gbatch, ptrue_rep):
    Nc = cval_gbatch.shape[1]
    bs = cval_gbatch.shape[0]
    
    gdterm = torch.zeros(Nc, device = cval_gbatch.device)
    for jc in range(Nc):        
        x = ptrue_rep.xvecs[jc]
        c = ds.constraints[:,jc]
        
        y = torch.zeros(x.shape[0], device=cval_gbatch.device)
        for s0 in pars['MMD_sig_vec']:
            s = s0/ptrue_rep.fsig_best[jc]
            y = y + torch.exp(log_prob_gaussian(x.unsqueeze(0), c.unsqueeze(1), s)).mean(0)/len(pars['MMD_sig_vec'])
        
        y2 = interp(x, y, cval_gbatch[:,jc])
        gdterm = gdterm + y2.sum(0)
           
    gdterm = - 2/bs**2*gdterm/len(pars['MMD_sig_vec'])
    return gdterm



def gather_f_interp(ds, ptrue_rep, kernel, sig_vec, device):
    f_interp_vec = []
    for jc in range(ds.Nc):
        x = ptrue_rep.xvecs[jc]
        c = ds.constraints[:,jc]
    
        #construct interpolating function
        y = torch.zeros(x.shape[0], device=device)
        for s0 in sig_vec:
            s = s0/ptrue_rep.fsig_best[jc]
            y = y + torch.exp(log_prob_gaussian(x.unsqueeze(0), c.unsqueeze(1), s)).mean(0)/len(sig_vec)
        f_interp_vec.append(get_f_interp(x, y))
    return f_interp_vec
    

def calculate_MMD_loss(pars, ds, ptrue_rep, cval_gbatch):    
    gterm = get_MMD_gterm(pars, cval_gbatch, ptrue_rep)
    gdterm = get_MMD_gdterm(pars, cval_gbatch, ptrue_rep)
    #gdterm = get_MMD_gdterm_v2(pars, ds, cval_gbatch, ptrue_rep)
    loss_kls = gterm + gdterm
    return loss_kls


