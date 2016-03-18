import numpy as np

def radloss(T):
    s = np.load('Q:\\python\\lib\\radloss.npz')
    logR = s['rlRate']
    logT = s['temperature']    
    logT1 = np.log10(T)    
    logR1 = np.interp(logT1, logT, logR, left = 0, right = 0)
    return 10**logR1