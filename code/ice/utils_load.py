# -*- coding: utf-8 -*-
from .utils import *


#%% data

def get_data_path(eval_opt='', E_opt='', lx=1024):
    dpath = get_folder_path('data')
    if lx == 896:
        return dpath + 'EM_grid_lowE.npz'
    if lx == 1024:
        return dpath + 'EM_1024_grid.npz'
    if lx == 4992 or lx == 5000: 
        return dpath + 'EM_4992_grid.npz'
    
    return path_filtered

def get_folder_path(opt='nets'):
    current_folder = os.path.basename(os.getcwd())
    if opt == 'nets':
        path = '../nets/' if current_folder == 'code' else '../../nets/'
    elif opt == 'data':
        path = '../data/' if current_folder == 'code' else '../../data/'   
    elif opt == 'plots':
        path = '../plots/' if current_folder == 'code' else '../../plots/'         
    return path



def load_corresponding_data(eval_opt, E_opt, lx = 896, normalize=False):
    path_filtered = get_data_path(eval_opt, E_opt, lx)
    x0, yn, a0, _ = load_data(path_filtered, Ns=-1, remove_invalid = False, normalize = normalize)
    if not normalize:
        yn = normalize_y(yn)
    if lx != x0.shape[1]:
        x0 = x0[:,-lx:]
    
    return x0, yn, a0


#%% amplitude classifier


#load amplitude classifer
def load_ac(path, name, device='cpu'):
    from .utils_ac import AmplitudeClassifier    
    
    with open(path + name + '.pars', 'rb') as f:
        pars = pickle.load(f)
    ac = AmplitudeClassifier(pars)
    ac.load(path + name + '.pt', device=device)
    ac.model = ac.model.to(device)    
    if not hasattr(ac, 'Nval'):
        ac.Nval = int(ac.ival.shape[0]/10)
    return ac



def load_standard_ac(seq_length=1024, device='cpu'):    
    #ac_path = '../nets/'
    ac_path = get_folder_path()
    if seq_length == 896:
        ac_name = 'ac_lowE'
    elif seq_length == 1024:
        ac_name = 'ac_May30'
    elif seq_length == 4992:
        ac_name = 'ac_5k_Jan05'
    ac = load_ac(ac_path, ac_name, device)
    ac.ac_name = ac_name
    return ac


#%% diffusion model

#load diffusion model
def load_ddpm(path, name, device='cpu', use_dataset = True):
    from .utils_ddpm import get_model
    from .utils import check_model_name, check_fpath
    name = check_model_name(name)
    path = check_fpath(path)

    if use_dataset == True:
        dataset = (torch.zeros(1), torch.zeros(1))
    else:
        dataset = -1
        
    par_name = name if name != 'final' else 'pars'
    with open(path + par_name + '.pars', 'rb') as f:
        pars = pickle.load(f)
        
    model, diffusion, trainer = get_model(dataset, pars, device)
    trainer.cpu = True
    diffusion.device = device
    diffusion.model.device = device
    diffusion.betas = diffusion.betas.to(device)
    trainer.load(path + name + '.pt')
    return model, diffusion, trainer




#%% theta transformation network

#load theta net
def load_TNet(path, name, device='cpu'):    
    from .utils_tnet import ThetaNet
    
    name = check_model_name(name)
    path = check_fpath(path)

    par_name = name if name != 'final' else 'pars'
    with open(path + par_name + '.pars', 'rb') as f:
        pars = pickle.load(f)
        pars['device'] = device
        
    if not 'weight_decay' in pars:
        pars['weight_decay'] = 0
    
    TNet = ThetaNet(pars)
    TNet.load(path + name +'.pt')
    TNet.model.to(device)
    return TNet

def get_standard_tnet_path(ds_opt):
    raise NotImplementedError()
    
    return model_path
     


#%% multiple models together


def load_models_for_evaluation(model_path, model_name, model_opt, seq_length=1024, device='cpu'):
    from ice.utils_eval import get_ppath

    ppath, model_name = get_ppath(model_path, model_name)
    
    if model_opt == 'ddpm':
        _, model, trainer = load_ddpm(model_path, model_name, device, use_dataset=False)
        model = model.to(device)
        model.is_ddim_sampling = True
        model.sampling_timesteps = 50
        trainer.pars['batch_signals'] = True
        model.batch_signals = True
        model.batch_with_noise = False
    elif model_opt == 'tnet':
        model = load_TNet(model_path, model_name, device = device)
    ac = load_standard_ac(seq_length, device)
    return model, ac, ppath
        