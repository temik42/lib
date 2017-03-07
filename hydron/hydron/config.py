import os
dir = os.path.dirname(__file__)
import numpy as np
from . import Sun


class Scale(object):
    def __init__(self, m=1, k_b=1, gamma=5./3, scale = [1.,1.,1.]):
        self.m = m
        self.k_b = k_b
        self.gamma = gamma
        self.set(scale)
        
    def set(self, scale):   
        self.t = scale[0]
        self.n = scale[1]
        self.T = scale[2]
        
        self.u = np.sqrt(self.k_b/self.m*self.T)
        self.x = self.u*self.t
        self.a = self.u/self.t
        
        self.rho = self.n*self.m
        self.rhou = self.rho*self.u
        self.rhoe = self.rhou*self.u
        
        self.rl = self.rhoe/self.n**2/self.t
        self.kappa = self.x**2/self.T*self.rhoe/self.t


class defconfig(object):
    def __init__(self):
        self.gamma = 5./3   #gamma-factor
        self.k_b = 1
        self.m = 1    
        self.base_scale = [1.,1.,1.]
        
        self.btype = 'mirror'   #boundary type; one of: 'continuous', 'mirror', 'periodic', 'constant' or 'stationary'
        self.itype = 'Roe'   #cell interface type; one of: 'average' or 'Roe'
        self.ctype = 'Riemann'   #correction-step type; one of: 'Riemann' or 'upwind'

        self.g = False   #acceleration, vector
        self.kappa = False   #diffusion coeffitient, function of T
        self.Lambda = False   #radiation loss function, function of T
        self.Hr = False   #heat rate function, function of s and t
        
        self.cfl_lim = 0.1
        self.wait = True
        self.verbose = True
        
        self.set_scale()
        
    def set_scale(self):
        self.scale = Scale(self.m,self.k_b,self.gamma,self.base_scale)
    

class loopconfig(defconfig):
    def __init__(self):
        defconfig.__init__(self)
        
        self.gamma = 5./3   #gamma-factor
        self.k_b = 2*Sun.k_b
        self.m = Sun.mu_c*Sun.m_p
        self.base_scale=[1.,Sun.n_c,Sun.T_c]
        
        self.btype = 'stationary'
        
        self.g = [0,0,-Sun.g_sun]
        self._Lambda = Sun.Lambda('chianti_cor.npz')
        self.Lambda = lambda T: self._Lambda.get(T)
        self.kappa = Sun.kappa
        
        self.set_scale()