import numpy as np
from scipy.spatial import Delaunay

class Bary(object):
    def __init__(self, X, I):
        self.X = X
        self.I = I
        self.tri = Delaunay(X.T).simplices
        
    def interp(self, Y):
        N = Y.shape[1]
        Yt = np.concatenate((Y, np.ones((1,N))), axis=0)
        J = np.zeros(N)
        for t in self.tri:
            A = np.linalg.inv(np.concatenate((self.X[:,t], np.ones((1,3))), axis=0))
            lam = np.dot(A,Yt)
            q = np.all([np.all(lam >= 0, axis = 0),np.all(lam <= 1, axis = 0)], axis=0)
            J[q] = np.dot(self.I[t],lam[:,q]) 
        return J