import numpy as np
from .config import defconfig


def Diffusion():
    N = 100       # number of gridpoints, units
    dt = 1e-3    # stepsize, s
    tau = 10  # total duration, s
    
    R = 1
    
    idx = np.arange(0,N,dtype=np.float32)/(N-1)
    X = idx*R
    
    n = np.ones(N, dtype = np.double)   # density, cm^{-3}
    u = np.zeros(N, dtype = np.double)     # velocity, cm/s
    T = np.ones(N, dtype = np.double)     # temperature, K
        
    cfg = defconfig()
    cfg.btype = 'constant'
    
    cfg.kappa = lambda T: 1e-2
    cfg.hrate = lambda s,t: 1e-2
    return (dt,tau,X,n,u,T,cfg)        


def Boltzmann():
    N = 100       # number of gridpoints, units
    dt = 1e-3    # stepsize, s
    tau = 100   # total duration, s
    
    R = 1
    #idx = np.arange(0,N,dtype=np.float32)/(N-1)
    #X = np.zeros((3,N), dtype = np.double)    # (N by 3) array of gridpoint coordinates, cm
    #X[0] = np.cos(np.pi*idx)*R              # x-coordinates, cm
    #X[2] = np.sin(np.pi*idx)*R              # y-coordinates, cm*L
    
    X = np.arange(0,N,dtype=np.float32)/(N-1)*R
    
    n = np.ones(N, dtype = np.double)   # density, cm^{-3}
    u = np.zeros(N, dtype = np.double)     # velocity, cm/s
    T = np.ones(N, dtype = np.double)     # temperature, K
        
    cfg = defconfig()
    cfg.g = [-1]
    cfg.btype = 'mirror'
    
    return (dt,tau,X,n,u,T,cfg)    

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
        
    cfg = defconfig()
    cfg.gamma = 7./5
    cfg.btype = 'mirror'
    
    return (dt,tau,X,n,u,T,cfg)