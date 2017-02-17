import numpy as np

def d(x, order = 1):
    if (order == 1):
        return np.gradient(x, edge_order = 1)
    if (order == 2):
        out = np.roll(x,1)+np.roll(x,-1)-2*x
        out[0] = out[1]
        out[-1] = out[-2]
        return out
    
def trisol(D,U,V,B):
    _D = np.array(D)
    _U = np.array(U)
    _V = np.array(V)
    _B = np.array(B)
    
    n = D.shape[0]
    
    for i in range(1,n):
        _D[i] -= _U[i-1]*_V[i-1]/_D[i-1]
        _B[i] -= _B[i-1]*_V[i-1]/_D[i-1]
        
    for i in range(n-2,-1,-1):
        _B[i] -= _B[i+1]*_U[i]/_D[i+1]
        
    return _B/_D