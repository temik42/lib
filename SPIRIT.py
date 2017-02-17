# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 11:02:25 2015

@author: artem
"""
import numpy as np

def Mg(T):
    x = np.arange(645, 805, 5)
    y = np.array([3,13,49,157,426,992,1999,3487,5276,6959,8107,8522,8287,
                 7639,6794,5912,5979,4330,3680,3125,2655,2260,1929,1650,1416,1218,
                 1051,910,790,687,599,523])   
    
    x1 = (np.log10(T)*100).clip(min = 645, max = 800)
    return np.interp(x1, x, y, left = 0, right = 0)*1e-47
    


        