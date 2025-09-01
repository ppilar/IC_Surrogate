import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import RegularGridInterpolator

from .utils import check_fpath
from .utils_load import *

#set parameters that have not been initialized before to standard values
def init_standard_pars_ac(pars, data_properties):
    from .utils import init_par
    pars['Ns'], pars['Nt'], pars['NE'], pars['Ni0'], pars['seq_length'] = data_properties
    init_par(pars, 'use_interpolation', False)
    
    init_par(pars, 'lr', 1e-4)
    init_par(pars, 'bs', 128)  
    init_par(pars, 'itmax', 150000)
    init_par(pars, 'itsched', 40000)    
    init_par(pars, 'fsched', 0.5)
    init_par(pars, 'fdrop', 0.)
    init_par(pars, 'pooling', 'avg')
    
    init_par(pars, 'conv_dims', [32, 64, 128])
    init_par(pars, 'embed_dims', [40, 40, 40])
    init_par(pars, 'flatten_out', 256)
    init_par(pars, 'classifier_dims', [256, 128, 1])
    init_par(pars, 'Nbin_interpolation', 10)
    
    init_par(pars, 'comment', 'bs='+str(pars['bs']))
    init_par(pars, 'run_dir', '../runs/runs_ac/')
    
    
    


def get_amplitude_hist2d_values(log_amplitudes, labels, Nbin = 20):
    counts, energy_bounds, theta_bounds, _ =  plt.hist2d(labels[:,0], labels[:,1], bins = Nbin)
    sums, _, _, _ =  plt.hist2d(labels[:,0], labels[:,1], weights = log_amplitudes, bins = Nbin)
    means = sums/counts
    var = np.zeros(sums.shape)
    for j in range(sums.shape[0]):
        for k in range(sums.shape[1]):
            buf, _, _, _ = plt.hist2d(labels[:,0], labels[:,1], weights = (log_amplitudes - means[j,k])**2, bins = Nbin);
            var[j,k] = buf[j,k]/counts[j,k]
    stds = np.sqrt(var)
    return counts, means, stds, energy_bounds, theta_bounds

def get_interpolating_functions(log_a, yn, pars):    
    counts, means, stds, energy_bounds, theta_bounds = get_amplitude_hist2d_values(log_a, yn, pars['Nbin_interpolation'])
    
    Emids = (energy_bounds[:-1] + energy_bounds[1:])/2
    tmids = (theta_bounds[:-1] + theta_bounds[1:])/2
    fmean = RegularGridInterpolator((Emids, tmids), means, bounds_error = False, fill_value = None)
    fstd = RegularGridInterpolator((Emids, tmids), stds, bounds_error = False, fill_value = None)    
    return fmean, fstd

def amplitudes_via_interpolation(log_a, yn, fmean):
    n_means = fmean((yn[:,0], yn[:,1]))
    log_a1 = log_a-n_means
    return log_a1


def complete_reverse_an(an, yn, a_bounds, fmean):
    from .utils import norm_mm
    log_a1 = norm_mm(an, a_bounds)
    n_means = fmean((yn[:,0], yn[:,1]))
    log_a = log_a1 + n_means
    return log_a, log_a1


#%% ac network
class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, mp_kernel_size=2, mp_stride=2, pooling='avg'):
        super(ConvBlock, self).__init__()
        
        if pooling == 'avg':
            self.pool = nn.AvgPool1d(kernel_size=mp_kernel_size, stride=mp_stride)
        elif pooling == 'max':
            self.pool = nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride)
        self.block = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='circular'),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride),
            self.pool
            )
    def forward(self, x):
        return self.block(x)
        
class CNN(nn.Module):
    def __init__(self, pars):
        from .utils import init_par

        super(CNN, self).__init__()
        self.embed_dim = pars['embed_dims'][-1]
        
        #some convolutional layers 
        if not 'pooling' in pars:
            pars['pooling'] = 'max'
        modules = []
        Cdims = [1] + pars['conv_dims']
        for j in range(len(pars['conv_dims'])):
            modules.append(ConvBlock(Cdims[j], Cdims[j+1], pooling=pars['pooling']))      
        self.features = nn.Sequential(*modules)        
        
        #flatten output
        self.flatten = nn.Sequential(            
            nn.Flatten(),
            nn.Linear(pars['conv_dims'][-1]*int(pars['seq_length']/2**len(pars['conv_dims'])), pars['flatten_out']),
            nn.ReLU(),
        )
        
        #embed E, t in high-D space
        embed_dims = [2] + pars['embed_dims']
        emodules = []
        for j in range(len(pars['embed_dims'])):
            emodules.append(nn.Linear(embed_dims[j], embed_dims[j+1]))
            if pars['fdrop'] > 0:
                emodules.append(nn.Dropout(p=pars['fdrop']))
            emodules.append(nn.ReLU())
        self.embedding = nn.Sequential(*emodules)
        
        #some fully connected layers
        cmodules = []
        classifier_dims = [pars['flatten_out'] + pars['embed_dims'][-1]] + pars['classifier_dims']
        for j in range(len(pars['classifier_dims'])):
            cmodules.append(nn.Linear(classifier_dims[j], classifier_dims[j+1]))
            if j < len(pars['classifier_dims'])-1:
                if pars['fdrop'] > 0:
                    cmodules.append(nn.Dropout(p=pars['fdrop']))
                cmodules.append(nn.ReLU())
        self.classifier = nn.Sequential(*cmodules)
        

    def forward(self, data):
        x = data[:,:,:-2]
        labels = data[:,:,-2:]
        
        x = self.features(x)
        x = self.flatten(x).unsqueeze(1)
        if self.embed_dim > 0:
            y = self.embedding(labels)            
            x = torch.cat((x,y),-1)
        x = self.classifier(x)
        return x.squeeze()
    
    

class AmplitudeClassifier(object):
    def __init__(
        self,
        pars = None,
        ):
        from .utils import init_par
        super().__init__()
        
        if pars is not None:     
            self.bs = pars['bs']
            self.lr = pars['lr']
            self.itmax = pars['itmax']
            self.fsched = pars['fsched']
            self.pars = pars
            self.initialize_model(pars)
            
        #model
        self.a_bounds = None
        self.fmean = None        
        
        self.it = 0
        self.epoch = 0
        
        self.criterion = nn.MSELoss()
        self.ival = None
        self.writer = None
        
    def initialize_model(self, pars):
        self.model = CNN(pars)
        self.optimizer = optim.Adam(self.model.parameters(), lr=pars['lr'])
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=pars['fsched'])
        
        
    def save(self, path):
        self.writer.close()
        data = {
            'it': self.it,
            'epoch': self.epoch,
            'bs': self.bs,
            'itmax': self.itmax,
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'sched': self.scheduler.state_dict(),
            'writer': self.writer,
            'ival': self.ival,
            'a_bounds': self.a_bounds,
            'fmean': self.fmean,
            'pars': self.pars
        }
        torch.save(data, path)
        
    
    def load(self, path, device='cpu'):
        data = torch.load(path, map_location=device)
        
        self.it = data['it']
        self.epoch = data['epoch']
        self.bs = data['bs']
        self.itmax = data['itmax']
        self.writer = data['writer']
        self.ival = data['ival']
        self.a_bounds = data['a_bounds']
        self.fmean = data['fmean']
        
        self.pars = data['pars']
        self.initialize_model(self.pars)
        
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['sched'])
        
    
    def initialize_interpolation(self, log_a, yn):
        from .utils import norm_mm
        self.fmean, _ = get_interpolating_functions(log_a, yn, self.pars)
        log_a1 = amplitudes_via_interpolation(log_a, yn, self.fmean)
        an, self.a_bounds = norm_mm(log_a1)  
        print('nans:',log_a1[np.isnan(log_a1)].shape)
        
    def normalize_input_amplitudes(self, log_a, yn):
        if self.pars['use_interpolation']:
            log_a = amplitudes_via_interpolation(log_a, yn[:,:2], self.fmean)
        return (log_a - self.a_bounds[0])/(self.a_bounds[1]-self.a_bounds[0])
    

        
    def reverse_normalization(self, an, yn):
        from .utils import norm_mm
        if self.pars['use_interpolation']:
            log_a, log_a1 = complete_reverse_an(an.detach(), yn.detach(), self.a_bounds, self.fmean)
        else:
            log_a = norm_mm(an, self.a_bounds)
            log_a1 = -1
        return log_a, log_a1
        
    def train_validation_split(self, data0):
        if self.ival is None:            
            #omit certain iN from data
            self.iN_val = 0
            self.Nval = torch.where(data0[2][:,2] == self.iN_val)[0].shape[0]
            self.ival = torch.sort(data0[2][:,2])[1]
        data = []
        for j in range(len(data0)):
            data.append(data0[0][self.ival])
            data.append(data0[1][self.ival])
            data.append(data0[2][:,:2][self.ival])
            
        inputs_train = torch.cat((data[0][self.Nval:].unsqueeze(1).float(), data[2][self.Nval:].unsqueeze(1).float()), -1)
        outputs_train = data[1][self.Nval:].float()
        yn_train = data[2][self.Nval:].float()
        
        inputs_val = torch.cat((data[0][:self.Nval].unsqueeze(1).float(), data[2][:self.Nval].unsqueeze(1).float()), -1)
        outputs_val = data[1][:self.Nval].float()
        yn_val = data[2][:self.Nval].float()

        return inputs_train, outputs_train, inputs_val, outputs_val
    
    
    def init_train(self, data, pars):
        from .utils import save_parameters, init_par, initialize_writer, norm_mm
        
        if self.pars['use_interpolation']:
            self.initialize_interpolation(data[1].cpu(), data[2][:,:2].cpu())
        else:
            _, self.a_bounds = norm_mm(data[1].cpu())
            
        init_par(self.pars, 'comment', '')
        self.writer = initialize_writer(self.pars['run_dir'], comment0 = self.pars['ds_opt'], comment = self.pars['comment'])
        
        self.pars['log_dir'] = self.writer.log_dir + '/'
        save_parameters(self.pars, self.pars['log_dir'])
        #print_parameters(self.pars, self.pars['log_dir'])

    def train(self, data, device='cpu'):
        from .plots import plot_amplitude_errors
        
        if self.a_bounds is None:
            self.init_train(data, self.pars)
            
                
                
        data[1] = self.normalize_input_amplitudes(data[1].cpu().clone(), data[2][:,:2].cpu())#.to(device)
        self.model = self.model.to(device)
        
        inputs_train, labels_train, inputs_val, labels_val = self.train_validation_split(data)#deepcopy(data))        
        train_dataset = TensorDataset(inputs_train, labels_train)
        del inputs_train, labels_train
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.bs, shuffle=True)
        
                    
        while self.it < self.itmax:
            for batch_inputs, batch_labels in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                
                #make prediction and calculate loss
                outputs = self.model(batch_inputs)  
                loss = self.criterion(outputs, batch_labels)
            
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.it%self.pars['itsched'] == 0 and self.it > 1:
                    self.scheduler.step()
                
                self.it += 1
                
            
                ## Model evaluation
                
                # Print training information
                print("\r" + f'Epoch [{self.epoch+1}], It: {self.it}, Loss: {loss.item():.4f}', end = "")
                        
                if self.it%20 == 0:
                    self.writer.add_scalar("loss", loss.detach().item(), self.it)
                    
                del outputs, loss
    
                
                with torch.no_grad():
                    if self.it%500 == 0 and self.it > 1: #evaluate on validation data
                        self.model.to('cpu')
                        outputs_val = self.model(inputs_val)
                        loss_validation = self.criterion(outputs_val, labels_val)
                        self.writer.add_scalar("validation_loss", loss_validation.detach().item(), self.it)
                        self.model.to(device)
                        
                        #self.model = self.model.to(device)
            
                    if self.it%5000 == 0 and self.it > 1: #save checkpoint
                        self.save('../nets/ac_test.pt')
                        
                    if self.it%10000 == 0 and self.it > 1: #make plot of the model performance (i.e. the error distribution)
                        log_a0r, _r = self.reverse_normalization(outputs_val.detach().cpu(), inputs_val[:,0,-2:].detach().cpu())
                        log_a0, _ = self.reverse_normalization(labels_val.detach().cpu(), inputs_val[:,0,-2:].detach().cpu())
                        fig = plot_amplitude_errors(log_a0, log_a0r)
                        self.writer.add_figure('amplitude_error_validation', fig, self.it)
                    
                    
                if self.it%500 == 0 and self.it > 1:
                    del outputs_val, loss_validation
        
                if self.it >= self.itmax:
                    break
            self.epoch += 1
        
        
        if self.itmax > 49999:
            self.save(self.pars['log_dir']+'ac_final.pt')
        
    
    def forward(self, data):             
        anr = self.model(data)
        log_ar, log_a1r = self.reverse_normalization(anr, data[:,:,-2:].squeeze())
        return log_ar, log_a1r, anr
        
        
        
        
#%%
def evaluate_ac(ac, data, ppath):
    ppath = check_fpath(ppath)
    
    print('')
    print('train test split')
    inputs_train, outputs_train, inputs_val, outputs_val = ac.train_validation_split(data)
    
    
    #%%
    print('overview!')
    evaluate_ac_overview(ac, inputs_train, outputs_train, ppath, dset = 'train')
    evaluate_ac_overview(ac, inputs_val, outputs_val, ppath, dset = 'val')
        
    
    #%% more detailed plots
    print('detailed evaluation')
    evaluate_ac_detailed(ac, [inputs_train.squeeze()[:,:-2], outputs_train, inputs_train.squeeze()[:,-2:]], ppath, 'train')
    evaluate_ac_detailed(ac, [inputs_val.squeeze()[:,:-2], outputs_val, inputs_val.squeeze()[:,-2:]], ppath, 'val')


def evaluate_ac_overview(ac, inputs, outputs, ppath, dset = 'train'):
    from .plots import plot_ac_error, plot_amplitude_comparison
    
    with torch.no_grad():
        ahat = ac.forward(inputs.detach())[0].cpu().numpy()    
    plot_ac_error(outputs.numpy(), ahat)
    plot_amplitude_comparison(outputs.numpy(), ahat, ppath = ppath, dset = dset)


def evaluate_ac_detailed(ac, data, ppath='', dset='train'):
    from .utils import get_Et_samples, get_Et_title
    
    xn, log_a0, yn = data
    device = xn.device
    
    Evec = [0.025, 0.25, 0.5, 0.75, 0.975]   
    tvec = Evec
    
    
    Np = len(Evec)    
    fig1, axs1 = plt.subplots(Np, Np, figsize=(3*Np, 3*Np))
    fig2, axs2 = plt.subplots(Np, Np, figsize=(3*Np, 3*Np))
    
    tol = 2.5e-2
    ac.model = ac.model.to(device)
    with torch.no_grad():
        for i, E in enumerate(Evec):
            for j, t in enumerate(tvec):
                print('i,j:',i,j)        
                xEt, yEt, log_aEt = get_Et_samples(xn, yn, log_a0, E, t, tol = tol)
                inputs = torch.cat((torch.tensor(xEt).unsqueeze(1), torch.tensor(yEt).unsqueeze(1)),-1).float().to(device)
                log_aEtr, _, _ = ac.forward(inputs.detach()) 
                
                print('Nr=',xEt.shape[0])
                error = log_aEt - log_aEtr.numpy()
                axs1[i,j].hist(error,101, range = (-0.5,0.5), density=True)
                axs1[i,j].set_xlabel(r'$\Delta$log10(a)')
                axs1[i,j].set_title(get_Et_title(E,t))
                
                axs2[i,j].hist(log_aEt, 101, density=True)
                axs2[i,j].set_xlabel(r'log10(a)')
                axs2[i,j].set_title(get_Et_title(E,t))
            
        
    fig1.subplots_adjust(hspace=0.6, wspace=0.3)
    fig1.savefig(ppath + dset + '_tol' + str(tol) + '_' + 'ac_evaluation.pdf', bbox_inches='tight')
    
    fig2.subplots_adjust(hspace=0.6, wspace=0.3)
    fig2.savefig(ppath + dset + '_tol' + str(tol) + '_' + 'amplitude_distribution.pdf', bbox_inches='tight')
    plt.show()