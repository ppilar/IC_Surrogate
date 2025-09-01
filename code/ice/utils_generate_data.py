# -*- coding: utf-8 -*-
from NuRadioMC.SignalGen import askaryan as ask
from NuRadioReco.utilities import units, fft
import numpy as np
import time

def generate_signals(parameters_scaled, Nsp=10, shower_type='EM', iN_vals=False, N_timebin = 896, dt = 1e-10):
    Nparameters = parameters_scaled.shape[0]
    
    # Define timebins and sampling rate
    N = N_timebin
    dt = dt * units.second # 1e-10 by default; 5e-10 would still be ok in terms of accuracy, but nice to have higher resolution
    sr = 1/dt
    
    
    if type(iN_vals) is not bool:
        Nsp = 1
        
    traces = np.zeros((Nparameters*Nsp, N+3))
        
    # Calculate the time traces of the signals
    tgen = time.time()
    for i in range(Nparameters):
        if np.mod(i,100) == 0:
            print(i, 'tgen=%f'%((time.time()-tgen)/(100*Nsp)))
            tgen = time.time()
        energy = parameters_scaled[i, 0]*units.eV
        theta = parameters_scaled[i, 1]*units.deg
        
        if type(iN_vals) is bool:
            for iN in range(Nsp):
                trace = ask.get_time_trace(
                    energy=energy,
                    theta=theta,
                    N=N,
                    dt=dt,
                    shower_type=shower_type, 
                    n_index=1.78,
                    R=1*units.km,
                    model="ARZ2020",
                    iN=iN
                )
                traces[i*Nsp + iN, 0] = energy
                traces[i*Nsp + iN, 1] = theta
                traces[i*Nsp + iN, 2] = iN
                traces[i*Nsp + iN, 3:] = trace
        else:
            trace = ask.get_time_trace(
                energy=energy,
                theta=theta,
                N=N,
                dt=dt,
                shower_type=shower_type, 
                n_index=1.78,
                R=1*units.km,
                model="ARZ2020",
                iN=iN_vals[i]
            )
            traces[i, 0] = energy
            traces[i, 1] = theta
            traces[i, 2] = iN_vals[i]
            traces[i, 3:] = trace
                
    
    
    # Filter signals by fourier transforming and setting frequencies above
    # the second peak (in frequency space) to zero.
    ff = np.fft.rfftfreq(N, dt)
    signals_filtered = np.zeros_like(traces[:,:])
    ff_cutoffs = np.zeros(traces.shape[0])
    for ind in range(traces.shape[0]):
    
        signal_spectrum = fft.time2freq(traces[ind,3:], sampling_rate=sr)
        freqs = np.abs(signal_spectrum)
            
        i = 0
        fmax = np.max(freqs)
        Nmax = 1
        while i+2 < freqs.shape[0]:
            if freqs[i] < fmax/10:
                if not Nmax == 3:
                    if freqs[i] < freqs[i+1] > freqs[i+2]:
                        Nmax += 1
                else:
                    if freqs[i] > freqs[i+1] < freqs[i+2]:
                        break
            i = i + 1
            
        j = 0
        j0 = -1
        Nx = traces[ind,3:].shape[0]
        amplitude = np.max(np.abs(traces[ind,3:]))
        j1 = Nx
        while j+10 < Nx:
            if np.abs(traces[ind,3+j]) >= amplitude/100 and j0 == -1:
                j0 = j
            if j0 != -1:
                if np.sum(np.abs(traces[ind,3+j:])) <= amplitude/100: 
                    j1 = j
                    break
            j = j+1
            
        #attenuate signals to zero away from main signal; required to avoid periodic boundaries in the generated signals
        attenuation = np.ones(Nx)
        attenuation *= np.exp(0.2*np.minimum(0, np.linspace(0,Nx-1,Nx) - j0))
        attenuation *= np.exp(0.2*np.minimum(0, j1 - np.linspace(0,Nx-1,Nx)))

    
        mask = ff < ff[i]
        signal_spectrum_filtered = np.zeros((signal_spectrum.shape), dtype=complex)
        signal_spectrum_filtered[mask] = signal_spectrum[mask]
        signal_filtered = fft.freq2time(signal_spectrum_filtered, sampling_rate=sr)
        signals_filtered[ind, 3:] = signal_filtered * attenuation
        signals_filtered[ind, :3] = traces[ind, :3]
        ff_cutoffs[ind] = i
    
    return signals_filtered, traces, Nsp, ff_cutoffs