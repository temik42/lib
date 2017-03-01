from . import Sun
from .radloss import RadLoss

class defconfig():
    def __init__(self):
        self.gamma = 5./3   #gamma-factor
        self.sctype = 'noscale'   #scale type; one of: 'coronal' or 'chromospheric'
        self.btype = 'mirror'   #boundary type; one of: 'continuous', 'mirror', 'periodic', 'constant' or 'ebtel'
        self.itype = 'average'   #cell interface type; one of: 'average' or 'Roe'
        self.h0 = 1
        self.k_b = 1
        self.m = 1
        self.g = [0]
        self.kappa = lambda T: 0
        self.Lambda = lambda T: 0
        self.hrate = lambda s,t: 0

class loopconfig(defconfig):
    def __init__(self):
        defconfig.__init__(self)
        self.sctype = 'coronal'   #scale type; one of: 'coronal' or 'chromospheric'
        cfg.btype = 'stationary'
        self.T0 = 2e4   #K, temperature on the boundary; only for ebtel boundary
        self.h0 = 1e7   #cm, height of transition region; only for ebtel boundary
        self.rl0 = 1.52   #radloss factor on the boundary; only for ebtel boundary
        self.k_b = 2*Sun.k_b
        self.m = Sun.mu_c*Sun.m_p
        self.g = [0,0,-Sun.g_sun]
        self.RL = RadLoss('radloss.npz')
        self.Lambda = lambda T: self.RL.get(T)
        self.kappa = lambda T: Sun.kappa*T**2.5