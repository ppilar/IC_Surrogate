import numpy as np
import torch
import os
import pickle
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

#initialize random seeds
def init_random_seeds(s=False):
    if type(s) == bool:
        s = s = np.random.randint(42*10**4)
        
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    
    rand_init = 1
    return rand_init, s


#initialize parameter value, if no value has been assigned previously
def init_par(pars, key, val):
    if not key in pars: pars[key] = val
    
    
def check_fpath(fpath):
    if fpath[-1] != '/':
        fpath += '/'
    return fpath

def check_model_name(name):
    if name[-3:] == '.pt':
        name = name[:-3]
    return name
    
def save_parameters(pars, fpath):
    fpath = check_fpath(fpath)
    print_parameters(pars, fpath)
    with open(fpath + 'pars.pars', 'wb') as f:
        pickle.dump(pars, f)
    
#print parameters
def print_parameters(pars, fpath):
    fpath = check_fpath(fpath)
    with open(fpath+'pars.txt', 'w') as f:
        for var in pars.keys():
            f.write(var+':'+str(pars[var])+'\n')
        f.write('\n\n')
        
#specify directory of writer
def initialize_writer(log_dir, comment0 = "", comment = ""):
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join(log_dir, comment0 + "_" + current_time + "_" + comment)
    return SummaryWriter(log_dir = log_dir)


def get_bool_array(N, inds):
    #returns array of N booleans; 
    #array_true: True for i in inds, else False
    #array_false: ~array_true
    
    array_true = torch.zeros(N).int()
    array_false = torch.ones(N).int()
    array_true[inds] = 1
    array_false[inds] = 0
    return array_true, array_false



def norm_mm(data, dbounds=False):
    if type(dbounds) is tuple: #reverse normalization
        dmin = dbounds[0]
        dmax = dbounds[1]
        return data*(dmax - dmin) + dmin
    else: #normalize to range [0,1]
        dmax = data.max()
        dmin = data.min()
        return (data - dmin)/(dmax - dmin), (dmin.item(), dmax.item())


def get_amplitudes(x):
    if type(x) == torch.Tensor:
        return torch.max(torch.abs(x),1)[0]
    else:
        return np.max(np.abs(x),1)

def remove_invalid_sequences(x, y, thr=0):
    amplitudes = get_amplitudes(x)
    ibuf = np.where(amplitudes <= thr)[0]
    x = np.delete(x, ibuf, 0)
    y = np.delete(y, ibuf, 0)    
    return x, y

def normalize_y(y):
    y[:,0] = normalize_E(y[:,0])
    y[:,1] = normalize_t(y[:,1])    
    return y

def normalize_E(E):
    Emin, Emax = get_Ebounds()
    return (E - Emin)/(Emax - Emin) 

def normalize_t(t):
    tmin, tmax = get_tbounds()
    return (t - tmin)/(tmax - tmin) 
    
def normalize_x(x):    
    amplitudes = get_amplitudes(x)
    if type(x) == torch.Tensor:
        x = ((x/(amplitudes.unsqueeze(1) + 1e-12)  + 1)/2.)          
    else:
        x = ((x/(amplitudes[..., None] + 1e-12)  + 1)/2.)  
    return x, amplitudes
    
def normalize_data(x, y):
    x, amplitudes = normalize_x(x)
    y = normalize_y(y)
      
    return x, y, amplitudes

def reverse_sample_normalization(x2, log_a):
    return (2*x2 - 1)*10**log_a.unsqueeze(1)

def get_Ebounds(): #bounds of log10(E[eV])
    return (15, 19)

def get_tbounds(): #bounds of theta [°]
    return (55.82 - 20, 55.82 + 20)


def get_tvec(N):
    if N == 1024:
        dt = 0.5
        return np.linspace(0, N*dt, N), 't [ns]'

def reverse_E_normalization(E):
    Emin, Emax = get_Ebounds()
    Ebuf = E*(Emax - Emin) + Emin
    return Ebuf

def reverse_t_normalization(t):
    tmin, tmax = get_tbounds()
    tbuf = t*(tmax - tmin) + tmin
    return tbuf

def convert_units(y, direction = 'forward'):    
    fac = 360/(2*np.pi)    
    if direction == 'forward':
        y[:,0] = np.log10(y[:,0])
        y[:,1] = y[:,1]*fac
    elif direction == 'backward':
        y[:,0] = 10**(y[:,0])
        y[:,1] = y[:,1]/fac        
    return y

def load_data(path, Ns=False, shorten_seq = False, remove_invalid = True, normalize = True):
    file = np.load(path)
    buf = 'data_properties' if 'data_properties' in file.keys() else 'arr_1'
    data_properties = file[buf]
    buf = 'data' if 'data' in file.keys() else 'arr_0'
    data = file[buf][:Ns,:] if type(Ns) == int else file[buf]
        
    
    i0 = 3 #first entries contain parameter values
    x = data[:, 345+i0:545+i0] if shorten_seq else data[:,i0:]        
    y = data[:,:3]    
    
    if remove_invalid:        
        x, y = remove_invalid_sequences(x, y)        
        
    y = convert_units(y)
    
    if normalize:
        x, y, a = normalize_data(x, y)
    else:        
        a = get_amplitudes(x)
        
    return x, y, a, data_properties
        

def get_Et_samples(samples, Et, amplitudes, E, t, tol=1e-4):
    Et_buf = Et.cpu().numpy() if type(Et) == torch.Tensor else Et

    iEbuf = np.where(np.abs(Et_buf[:,0] - E) < tol)[0]
    samples = samples[iEbuf]
    Et = Et[iEbuf]
    Et_buf = Et_buf[iEbuf]
    amplitudes = amplitudes[iEbuf]
    
    itbuf = np.where(np.abs(Et_buf[:,1] - t) < tol)[0]
    samples = samples[itbuf]
    Et = Et[itbuf]
    amplitudes = amplitudes[itbuf]
    return samples, Et, amplitudes


def get_xEt_gen(diffusion, ac, E, t, N, tol = 0, normalize=True):
    from .utils_ddpm import sample_from_ddpm
    
    device = diffusion.device
    E = torch.tensor(np.random.uniform(max(E-tol,0),min(E+tol,1),(N,1,))).float().to(device)
    t = torch.tensor(np.random.uniform(max(t-tol,0),min(t+tol,1),(N,1,))).float().to(device)
    classes = torch.cat((E,t),1).float().to(device)
    
    Et_samples_gen, _ = sample_from_ddpm(diffusion, N, classes=classes)
    if not normalize:
        log_ar, log_a1r, anr = ac.forward(torch.cat((Et_samples_gen, classes),-1).unsqueeze(1))
        Et_samples_gen = reverse_sample_normalization(Et_samples_gen, log_ar)
    else:
        Et_samples_gen = Et_samples_gen
    return Et_samples_gen, classes


def get_Et_title(E,t):
    return ('E = %.2eeV'%(10**reverse_E_normalization(E))+r', $\theta$=%.2f°'%(reverse_t_normalization(t)))


def get_E_title(E):
    return ('E = %.2eeV'%(10**reverse_E_normalization(E)))


def extract_metrics(x, mlist):
    if type(mlist) != list:
        mlist = [mlist]
    Nm = len(mlist)
    Nx = x.shape[0]
    
    metrics = torch.zeros(Nm, Nx)
    for jm, metric in enumerate(mlist):
        if metric == 'minmax':
            metrics[jm] = torch.max(x,1)[0] - torch.min(x,1)[0] + 1e-12
        if metric == 'amplitude':
            metrics[jm] = torch.max(torch.abs(x),1)[0] + 1e-12
        if metric == 'Efluence':
            metrics[jm] = torch.trapz(torch.abs(x)**2, dim=1) + 1e-12
            
    return metrics
