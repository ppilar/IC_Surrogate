from NuRadioMC.SignalGen import askaryan as ask
from NuRadioReco.utilities import units, fft
import numpy as np
import time
import sys
from ice.utils_generate_data import generate_signals

#%% settings
if len(sys.argv) > 1:
    gen_mode = sys.argv[1]
else:
    gen_mode = 'grid' #'random'
    
print('gen_mode:', gen_mode)
if gen_mode not in ['random', 'eval', 'grid', 'test_dE']:
    print('Please select valid gen_mode!')



shower_type = "EM"  #"had", "EM"

N_timebin = 1024
dt = 5e-10 #timebin in [seconds]

#Nsamples needs to be multiples of the number of shower profiles, Nsp=10
Nsamples = 100000 #used for random generation
Nsamples_per_parameter = 5000 #used for eval generation

#bounds
Emin = 15 #log10(E[eV])
Emax = 19 #log10(E[eV])
tmin = 55.82 - 20 #[°]
tmax = 55.82 + 20 #[°]


#misc
eps = 2.5e-2 #spread around (normalized) selected parameter values in case of 'eval'
fname = shower_type + '_' + str(N_timebin) #name of file


#%% generate parameter arrays
Nsp = 10 #number of shower profiles (are there always exactly 10?)
data_properties = [Nsamples, -1, -1, Nsp, N_timebin]
if gen_mode == 'random': #generate data with random values (E,t) for training
    Nparameters = int(Nsamples/Nsp)
    samples = np.random.random_sample((Nparameters, 2))
    Evals_ges = samples[:,0]*(Emax - Emin) + Emin
    tvals_ges = samples[:,1]*(tmax - tmin) + tmin
    
if gen_mode == 'grid': #generate data on grid (E,t) for training
    print(gen_mode)
    dE = 0.1 #0.05 #energy bins s.t. different shower profiles
    NE = int((Emax - Emin)/dE+1)*3 #*3 in order to get 3 samples per energy bin; some amplitude dependence part of NuRadioMC
    Nt = 160#161#41
    Evals_ges = np.linspace(Emin, Emax, NE).repeat(Nt,-1)
    tvals_ges = np.linspace(tmin, tmax, Nt+1)[:-1]
    tvals_ges = np.expand_dims(tvals_ges,1).repeat(NE,1).T.reshape(-1)
    
    fname += '_grid3'
    Nparameters = NE*Nt
    data_properties = [Nt*NE*Nsp, Nt, NE, Nsp, N_timebin]
if gen_mode == 'test_dE':
    dE = 0.001
    NE = int((Emin + 0.5 - Emin)/dE)
    Nt = 5
    Evals_ges = np.linspace(Emin, Emin + 0.5, NE).repeat(Nt,-1)
    tvals_ges = np.expand_dims(np.linspace(tmin, tmax, Nt),1).repeat(NE,1).T.reshape(-1)
    
    fname += '_test_dE_v2'
    Nparameters = NE*Nt
    
if gen_mode == 'eval': #generate data around specific values (E,t) for precise evaluation (will require more shower profiles)
    #normalized values (E,t)
    Evals = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    tvals = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    N0 = int(Nsamples_per_parameter/Nsp)
    
    
    Evals_ges = np.repeat(np.repeat(Evals[np.newaxis], len(tvals),0).flatten(), N0)
    tvals_ges = np.repeat(np.repeat(tvals[np.newaxis], len(Evals),1).flatten(), N0)
    Enoise = np.random.uniform(-eps,eps,Evals_ges.shape)
    tnoise = np.random.uniform(-eps,eps,tvals_ges.shape)
    Evals_ges = (Evals_ges + Enoise)*(Emax - Emin) + Emin
    tvals_ges = (tvals_ges + tnoise)*(tmax - tmin) + tmin
    
    Nparameters = N0*len(Evals)*len(tvals)    
    fname += '_eps'+str(eps)


parameters_scaled = np.zeros((Nparameters, 2))
parameters_scaled[:,0] = 10**Evals_ges
parameters_scaled[:,1] = tvals_ges
        
    
#%% data generation code
signals_filtered, _, _, _ = generate_signals(parameters_scaled, Nsp=Nsp, shower_type=shower_type, N_timebin=N_timebin, dt=dt)

#sort signals s.t. the same signal for different angles is stored in subsequent elements of the array
if gen_mode == 'grid': #TODO: check
    x = signals_filtered[:,3:]
    y = signals_filtered[:,:3]
    lx = x.shape[1]
    xbatch = np.transpose(x.reshape(-1,10,lx),(1,0,2)).reshape(-1,lx)
    ybatch = np.transpose(y.reshape(-1,10,3),(1,0,2)).reshape(-1,3)
    signals_filtered = np.concatenate((ybatch, xbatch), 1)

np.savez(f'../data/'+ fname, data = signals_filtered.astype('float32'), data_properties = data_properties)
