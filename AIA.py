# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 11:02:25 2015

@author: artem
"""
import numpy as np

def t171(T):
    x = np.arange(500, 660, 5)
    y = np.array([15, 13, 12, 16, 29, 66, 137, 247, 381, 501, 647, 1053, 
                  1973, 3538, 5717, 8301, 10905, 13017, 14090, 13671, 11659, 8515, 5178, 2555,
                  1020, 384, 238, 241, 204, 111, 43, 18])   
    
    x1 = (np.log10(T)*100).clip(min = 500, max = 655)
    return np.interp(x1, x, y, left = 0, right = 0)*1e-28
    

def t193(T):
    x = np.arange(500, 700, 5)
    y = np.array([23, 22, 23, 30, 57, 126, 258, 449, 646, 735, 695, 678, 733, 816, 895,
                  966, 1033, 1104, 1204, 1398, 1829, 2678, 3948, 5102, 5173, 3790, 1894,
                  677, 227, 102, 69, 60, 56, 57, 62, 63, 54, 40, 28, 21])   
    
    x1 = (np.log10(T)*100).clip(min = 500, max = 695)
    return np.interp(x1, x, y, left = 0, right = 0)*1e-28
    

def t131(T):
    x = np.arange(500, 675, 5)
    y = np.array([4, 6, 9, 11, 15, 20, 34, 85, 256, 674, 1369, 2225, 3069, 3725, 4057,
                  4000, 3604, 3018, 2358, 1699, 1116, 668, 377, 218, 147, 119, 106, 95, 
                  86, 79, 73, 67, 61, 57, 56])   
    
    x1 = (np.log10(T)*100).clip(min = 500, max = 670)
    return np.interp(x1, x, y, left = 0, right = 0)*1e-29

def t211(T):
    x = np.arange(500, 700, 5)
    y = np.array([264, 271, 311, 408, 580, 829, 1114, 1378, 1565, 1557, 1380, 1249, 1217, 1243, 1319, 
                  1462, 1688, 1975, 2259, 2483, 2702, 3193, 4499, 7392, 12071, 16216, 15503, 10162,
                  5155, 2379, 1118, 589, 385, 313, 269, 210, 147, 100, 71, 53])   
    
    x1 = (np.log10(T)*100).clip(min = 500, max = 695)
    return np.interp(x1, x, y, left = 0, right = 0)*1e-29
   

def tresp(T, channel):
    if channel == 1 or channel == 'A131': return t131(T)
    if channel == 2 or channel == 'A171': return t171(T)
    if channel == 3 or channel == 'A193': return t193(T)
    if channel == 4 or channel == 'A211': return t211(T)
        