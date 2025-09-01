# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from .utils import get_Et_title, get_tvec

plt.rcParams['font.size'] = 14  # Set global font size
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Helvetica']  # Optional: specific font
plt.rcParams['figure.dpi'] = 300
plt.rcParams['text.usetex'] = True  # Use LaTeX for rendering text


def plot_amplitude_errors(a1, a1r):
    errors = torch.log10(a1r) - torch.log10(a1)
    
    fig = plt.figure()
    plt.hist(errors,1000,density=True)
    plt.xlim(-.5, .5)
    plt.xlabel(r"$\Delta$ log(a)")
    plt.title('NN error')
    return fig


#plot a number of dataset samples
def plot_samples(samples, spath=''):
    if type(samples) == tuple:
        dscores = samples[1]
        samples = samples[0]
    else:
        dscores = -1
    sbuf = samples.detach().squeeze().cpu().numpy()
    
    #### 4x4 plots    
    Np = 5
    fig, axs = plt.subplots(Np,Np, figsize=(3*Np,3*Np))
    for j in range(Np**2):
        if sbuf.ndim == 2:
            axs[j//Np, j%Np].plot(sbuf[j])
        if sbuf.ndim == 3:
            axs[j//Np, j%Np].imshow(sbuf[j])
                   
    plt.show()    
    return fig


def plot_amplitude_comparison_v1(ages):
    if type(ages[0]) is tuple:
        Nr = len(ages)
        Na = len(ages[0])
    else:
        Na = len(ages)
        ages = (ages,)
        Nr = 1
        
    fig, axs = plt.subplots(Na, Nr, figsize = (4.5*Nr,4*Na))
    print(Nr)
    print(Na)
    
    l = ['unnormalized', 'via interpolation', 'interpolation + norm']
    lr = ['real amplitudes', 'real amplitudes via NN', 'generated amplitudes']
    
    xlim = [(-7,1), (-3,3), (-2,2)]
    ylim = [(0,0.6), (0,3), (0, 25)]#
    for jr in range(Nr):
        a = ages[jr]
        for j in range(Na):
            if Nr > 1:
                ax = axs[j, jr] 
            else:
                ax = axs[jr]
            
            ax.hist(np.log10(a[j]), 100, density = True)
            #ax.text(0.55,0.9,l[j], fontsize = 12, transform=ax.transAxes)
            ax.set_xlim(xlim[j])
            ax.set_ylim(ylim[j])
            ax.set_xlabel('log(a)')
            if j == 0:
                ax.set_title(lr[jr], fontsize=14)
                #axs[jr,j].text(-3,0.55,lr[jr], fontsize = 14)
    
    fig.subplots_adjust(hspace=0.3)
    return fig
    
    
def plot_amplitude_comparison_v0(ages):
    if type(ages[0]) is tuple:
        Nr = len(ages)
        Na = len(ages[0])
    else:
        ages = (ages,)
        Nr = 1
        Na = len(ages)
        
    fig, axs = plt.subplots(Nr, Na, figsize = (4.5*Na,4*Nr))
    
    l = ['unnormalized', 'via interpolation', 'interpolation + norm']
    lr = ['real amplitudes', 'real amplitudes via NN', 'generated amplitudes']
    
    xlim = [(-6,1), (-3,3), (-2,2)]
    ylim = [(0,0.6), (0,3), (0, 25)]#
    for jr in range(Nr):
        a = ages[jr]
        for j in range(Na):
            axs[jr,j].hist(np.log10(a[j]), 100, density = True)
            axs[jr,j].text(0.55,0.9,l[j], fontsize = 12, transform=axs[jr,j].transAxes)
            axs[jr,j].set_xlim(xlim[j])
            axs[jr,j].set_ylim(ylim[j])
            if j == 0:
                axs[jr,j].set_title(lr[jr], fontsize=14)
    
    fig.subplots_adjust(hspace=0.3)
    return fig



def axplot_value_dist(ax, vals, bounds, labels):
    if type(vals) not in [tuple, list]:
        vals = [vals]
    for j, val in enumerate(vals):  
        axplot_hist_ncounts(ax, val, 51, xlims=bounds, alpha=0.7, label=labels[j])
    ax.set_title(r'$\log(a) - \log(\hat a)$')
    ax.legend()
    
def axplot_error_dist(ax, errors, bounds = (-0.1, 0.1), labels = ['train', 'test']):
    axplot_value_dist(ax, errors, bounds, labels)
    ax.set_title('error distribution')
    ax.set_xlabel(r'$\Delta$log10(a)')
    ax.legend()
    
def axplot_amplitude_dist(ax, errors, bounds = (-8, 1), labels = ['train', 'test']):
    axplot_value_dist(ax, errors, bounds, labels)
    ax.set_title('real amplitudes')
    ax.set_xlabel('log10(a)')
    ax.legend()


    
def axplot_Et_envelope(ax, x, x_gen, y, y_gen, E, t, tol = 1e-4, opt='samples'):    
    xEt = torch.tensor(get_Et_samples(x, y.numpy(), E, t, tol = tol))
    xEt_gen = torch.tensor(get_Et_samples(x_gen, y_gen.numpy(), E, t, tol = tol))    
    axplot_sample_comparison(ax, xEt, xEt_gen, E, t, opt = opt)
    print(xEt.shape[0])
    print(xEt_gen.shape[0])
    return xEt, xEt_gen


def axplot_sample_comparison(ax, Et_samples, Et_samples_gen, E, t, opt):
    i0, i1, sq0, sq3 = axplot_envelope(ax, Et_samples)
    tvec, tlabel = get_tvec(Et_samples.shape[-1])
    if opt == 'contour':
        _, _, gq0, gq3 = axplot_envelope(ax, Et_samples_gen, inds=(i0, i1), opt=opt)        
        if torch.max(torch.abs(Et_samples_gen)) > 5*torch.max(sq3):
            ax.set_ylim([-5*torch.max(sq3).item(), 5*torch.max(sq3).item()])        
        
    if opt == 'samples':
        axplot_samples(ax, Et_samples_gen[:,i0:i1], xrange = tvec[i0:i1], linestyle='--')
    ax.set_title(get_Et_title(E,t))
    ax.set_xlabel(tlabel)

    

def get_xrange(samples):
    maxs = torch.max(samples, 0)[0]
    mins = torch.min(samples, 0)[0]
    
    Nx = samples.shape[1]    
    buf = torch.maximum(torch.abs(maxs), torch.abs(mins))
    buf_max = buf.max().item()
    for j in range(Nx):
        if torch.abs(buf[j]) >= buf_max/50:
            #i0 = j
            i0 = np.max((j-5, 0))
            break
    for j in reversed(range(Nx)):
        if torch.abs(buf[j]) >= buf_max/50:
            #i1 = j
            i1 = np.min((j+5, Nx-1))
            break            
    
    Nx = i1-i0
    if Nx < 20:
        di = int(np.ceil((20-Nx)/2))
        i0 = i0-di
        i1 = i1+di
        Nx = i1-i0
        
    return i0, i1

def axplot_envelope(ax, samples, inds = False, opt='fill'):
    maxs = torch.max(samples, 0)[0]
    mins = torch.min(samples, 0)[0]
    
    q0 = torch.quantile(samples, .1, 0)
    #q1 = torch.quantile(samples, .25, 0)
    #q2 = torch.quantile(samples, .75, 0)
    q3 = torch.quantile(samples, .9, 0)
    
    tvec, tlabel = get_tvec(samples.shape[-1])
    
    if type(inds) != bool:
        i0, i1 = inds
    else:
        i0, i1 = get_xrange(samples)
        
    
    #xbuf = range(i0,i1)
    xbuf = tvec[i0:i1]
        
    if opt == 'fill':
        ax.fill_between(xbuf, mins[i0:i1], maxs[i0:i1], facecolor='blue', alpha=.2)
        ax.fill_between(xbuf, q0[i0:i1], q3[i0:i1], facecolor='blue', alpha=.5)
        buf = max(abs(q0.min().item()), abs(q3.max().item()))
        
    if opt == 'contour':
        ax.plot(xbuf, mins[i0:i1], 'k--', alpha=0.5)
        ax.plot(xbuf, maxs[i0:i1], 'k--', alpha=0.5)
        ax.plot(xbuf, q0[i0:i1], 'k--')
        ax.plot(xbuf, q3[i0:i1], 'k--')
        
    return i0, i1, q0, q3
    
    
def axplot_samples(ax, samples, linestyle='--', xrange = False):
    if type(xrange) != bool:
        ax.plot(xrange, samples.T, linestyle, linewidth = 0.5)
    else:
        ax.plot(samples.T, linestyle, linewidth = 0.5)
        
        
def plot_means(means, energy_bounds, theta_bounds):
    plt.imshow(means, extent = (energy_bounds[0], theta_bounds[0], energy_bounds[1], theta_bounds[1]))
    plt.colorbar()
    
    
def plot_amplitude_errors(log_a1, log_a1r):
    errors = log_a1r - log_a1
    
    fig = plt.figure()
    plt.hist(errors,1000,density=True)
    plt.xlim(-.5, .5)
    plt.xlabel(r"$\Delta$ log(a)")
    plt.title('NN error')
    return fig


# interpolation plot
def plot_interpolation(a0, y0):
    y0_buf = y0.clone()
    y0_buf[:,1] = y0_buf[:,1]/(2*torch.pi)*360
    means, sums, counts, eb, tb = get_hist2d_means(a0, y0_buf)
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(np.flip(means,1), extent = (eb[0], eb[-1],tb[0], tb[-1]), aspect = (eb[-1]-eb[0])/(tb[-1]-tb[0]))
    ax.set_xlabel('log(E [eV])')
    ax.set_ylabel(r'$\theta [^\circ]$')
    ax.set_title('log(a)')
    fig.colorbar(im, ax=ax)
    fig.savefig(ppath + 'interpolation.pdf', bbox_inches='tight')
    
def axplot_hist_ncounts(ax, vals, bins, xlims, alpha, label=''):
    counts, bin_edges = np.histogram(vals, bins=bins, range=xlims)
    normalized_counts = counts / vals.shape[0]  # sum of heights = 1
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # Plot normalized counts as bars
    ax.bar(bin_centers, normalized_counts, width=np.diff(bin_edges), edgecolor=None, alpha=alpha, label=label)
    #print(normalized_counts.sum())
    
def axplot_metric(ax, metric, metric_gen, E, t, metric_label, ks_ges, w1_ges):
    hmin = min(np.log10(metric.min()), np.log10(metric_gen.min())).item()
    hmax = max(np.log10(metric.max()), np.log10(metric_gen.max())).item()
    mrange = (hmin, hmax)
    counts, bins, _ = ax.hist(torch.log10(metric), bins=50, density=True, histtype='step', label='true', range=mrange)
    ax.hist(torch.log10(metric_gen), bins = bins, density=True, label='gen')#, range=mrange)            
    
    ax.set_title(get_Et_title(E,t))
    ax.set_xlabel(metric_label)
    
    mean_error = torch.abs(metric.mean() - metric_gen.mean())/metric.mean()
    std_error = torch.abs(metric.std() - metric_gen.std())/metric.std()    
    
    ks = stats.kstest(metric, metric_gen)
    w1 = stats.wasserstein_distance(torch.log10(metric), torch.log10(metric_gen))
    ks_ges.append(ks)
    w1_ges.append(w1)  
    
    ax.text(0.01, 0.99, 
            'pKS=%.3f '%(ks[1]) + '\n' 
            + 'W1=%.3f'%(w1) + '\n'
            #+ 'W1_0=%.3f'%(w1_0) + '\n'
            + r'$\Delta \mu=%.1f$'%(mean_error*100) + '%' + '\n'
            + r'$\Delta \sigma=%.1f$'%(std_error*100) + '%',
             horizontalalignment='left',
             verticalalignment='top',
             transform = ax.transAxes)
    
    return ks, w1


def axplot_real_gen_samples(axr, axg, xEt, xEt_gen, E, t):
    i0, i1 = get_xrange(xEt)
    tvec, tlabel = get_tvec(xEt.shape[-1])
    axr.plot(tvec[i0:i1], xEt[:10,i0:i1].T, linewidth=0.75)  
    axg.plot(tvec[i0:i1], xEt_gen[:100,i0:i1].T, linewidth=0.75)
    axr.set_title(get_Et_title(E,t) + ', simulated samples')
    axg.set_title(get_Et_title(E,t) + ', generated samples')
    axg.set_ylim(axr.get_ylim())
    axr.set_xlabel(tlabel)
    axg.set_xlabel(tlabel)
    
    
def plot_ac_error(log_a0, log_ahat):
    fig, axs = plt.subplots(1, 3, figsize=(13,4))
    
    error = np.abs(np.log10(10**log_a0 - 10**log_ahat))/log_a0
    axs[0].hist(error, 51, range = (-4, 1.), density=True)
    axs[0].set_title(r'$|\log(a-\hat a)|/\log(a)$')
    
    error = np.abs(10**log_a0 - 10**log_ahat)/10**log_a0
    axs[1].hist(error, 51, range = (0, 1.2), density=True)
    axs[1].set_title(r'$|a - \hat a|/a$')
    
    error = log_a0 - log_ahat
    axs[2].hist(error, 51, range = (-0.5, 0.5), density=True)
    axs[2].set_title(r'$\log(a) - \log(\hat a)$')
    
def plot_amplitude_comparison(log_a0, log_ahat, ppath, dset='train'):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(log_a0, 101, density=True, range = (-8,1))
    axs[0].set_title('real amplitudes (' + dset + ')')
    axs[0].set_xlabel('log10(a)')
    error = log_ahat - log_a0
    axs[1].hist(error, 101, density=True, range=(-0.3,0.3))
    axs[1].set_title('error in amplitude predictions (' + dset + ')')
    axs[1].set_xlabel(r'$\Delta$log10(a)')    
    plt.savefig(ppath + dset +  '_amplitude_comparison.pdf', bbox_inches='tight')
    
    
    
    
def plot_true_and_preprocessed(x0, x1, y0, y1):
    plt.figure()
    plt.plot(x0, label='orig')
    plt.plot(x1, label='true')
    xtransf = transform_signal(x0,y0,y1)
    plt.plot(xtransf, label='transf')
    plt.legend()
    
    
def plot_interpolation(samples, bs, legend = ''):
    fig, axs = plt.subplots(bs, 1, figsize=(16,3*bs))
    for j in range(bs):
        axs[j].plot(samples[j::bs].T)
        if type(legend) == list and j == bs-1:
            axs[j].legend(legend)
            
    return fig
