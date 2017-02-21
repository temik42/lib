import numpy as np
from .config import defconfig


def Sod_tube():
    N = 1000       # number of gridpoints, units
    dt = 1e-4    # stepsize, s
    tau = 0.1   # total duration, s
    
    L = 1
    X = np.arange(0,N,dtype=np.float32)/(N-1)*L
    
    n = np.ones(N, dtype = np.double)*1.25   # density, cm^{-3}
    n[0:N/2] = 10
    u = np.zeros(N, dtype = np.double)     # velocity, cm/s
    T = np.ones(N, dtype = np.double)*1     # temperature, K
    T[0:N/2] = 1.25
        
    config = defconfig()
    config.gamma = 7./5
    
    hrate=lambda s,t:0
    return (dt,tau,X,n,u,T,hrate,config)