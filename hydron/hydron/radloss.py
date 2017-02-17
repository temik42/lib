import os,sys
dir = os.path.dirname(__file__)
import numpy as np

class RadLoss():
    def __init__(self, rl_fname='radloss.npz'):
        rl = np.load(dir+'\\'+rl_fname)
        self.rlRate = 10**rl['rlRate']
        self.rlTemperature = 10**rl['temperature']
    
    def get(self,T):
        def left(T):
            return 1.16e-31*T**2
        
        return np.where(T > 1e4, np.interp(T, self.rlTemperature, self.rlRate, left = left(1e4), right = 0),
                        left(T))