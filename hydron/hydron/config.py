from . import Sun

class defconfig():
    def __init__(self):
        self.gamma = 5./3   #gamma-factor
        self.sctype = 'noscale'   #scale type; one of: 'coronal' or 'chromospheric'
        self.btype = 'mirror'   #boundary type; one of: 'continuous', 'mirror', 'periodic', 'constant' or 'ebtel'
        self.itype = 'average'   #cell interface type; one of: 'average' or 'Roe'
        self.k_b = 1
        self.m = 1
        self.g = 1
        self.downvec = [0,0,-1]   #direction of gravity

class loopconfig(defconfig):
    def __init__(self):
        defconfig.__init__(self)
        self.sctype = 'coronal'   #scale type; one of: 'coronal' or 'chromospheric'
        self.T0 = 2e4   #K, temperature on the boundary; only for ebtel boundary
        self.h0 = 1e7   #cm, height of transition region; only for ebtel boundary
        self.rl0 = 1.52   #radloss factor on the boundary; only for ebtel boundary
        self.k_b = 2*Sun.k_b
        self.m = Sun.mu_c*Sun.m_p
        self.g = Sun.g_sun