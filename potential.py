import numpy as np
from scipy import fftpack as fft

def dst2d(X):
    return fft.dst(fft.dst(X,1, axis = 1),1,axis = 0)


def idst2d(X):
    N = X.shape
    return fft.idst(fft.idst(X,1, axis = 1),1, axis = 0)/(4*(N[0]+1)*(N[1]+1))


def PField(X, nz=1, scale=1):
    dim = X.shape
    x = np.mgrid[0:dim[0], 0:dim[1], 0:nz].astype(np.float64)
    return idst2d(np.exp(-2*np.pi*x[2]*scale*np.sqrt((x[0]/dim[0])**2 + (x[1]/dim[1])**2))*
                    (np.resize(dst2d(X),(nz, dim[0],dim[1])).transpose((1,2,0))))