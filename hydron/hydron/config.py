from . import Sun
from .radloss import RadLoss

class defconfig():
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

class loopconfig(defconfig):
    def __init__(self):
        defconfig.__init__(self)
        self.sctype = 'coronal'
        self.btype = 'stationary'
        self.T0 = 2e4   #K, temperature on the boundary
        self.k_b = 2*Sun.k_b
        self.m = Sun.mu_c*Sun.m_p
        self.g = [0,0,-Sun.g_sun]
        self.RL = RadLoss('radloss.npz')
        self.Lambda = lambda T: self.RL.get(T)
        self.kappa = lambda T: Sun.kappa*T**2.5