import numpy as np
from . import Sun

class Scale(object):
    def __init__(self, m=1, k_b=1, sctype = 'noscale'):
        self.m = m
        self.k_b = k_b
        
        if (sctype == 'noscale'):
            self.set(1.,1.,1.)
            
        if (sctype == 'coronal'):
            self.set(1.,1e9,1e6)
        
        if (sctype == 'chromospheric'):
            self.set(1.,1e11,1e4)   
        
    def set(self, t, n, T):   
        self.t = t
        self.n = n
        self.T = T
        
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
        
        self.sctype = 'noscale'   #scale type; one of: 'noscale', 'coronal' or 'chromospheric'
        self.btype = 'mirror'   #boundary type; one of: 'continuous', 'mirror', 'periodic', 'constant' or 'stationary'
        self.itype = 'Roe'   #cell interface type; one of: 'average' or 'Roe'
        self.ctype = 'Riemann'   #correction-step type; one of: 'Riemann' or 'upwind'

        self.g = False   #acceleration, vector
        self.kappa = False   #diffusion coeffitient, function of T
        self.Lambda = False   #radiation loss function, function of T
        self.Hr = False   #heat rate function, function of s and t
        
        self.wait = True
        self.verbose = True
        
        self.set_scale()
        
    def set_scale(self):
        self.scale = Scale(self.m,self.k_b,self.sctype)
    

class loopconfig(defconfig):
    def __init__(self):
        defconfig.__init__(self)
        
        self.gamma = 5./3   #gamma-factor
        self.k_b = 2*Sun.k_b
        self.m = Sun.mu_c*Sun.m_p
        
        self.sctype = 'coronal'
        self.btype = 'stationary'
        
        self.g = [0,0,-Sun.g_sun]
        self.RL = Sun.RadLoss('radloss.npz')
        self.Lambda = lambda T: self.RL.get(T)
        self.kappa = Sun.kappa
        
        self.set_scale()