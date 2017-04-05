import numpy as np


pi=np.pi

c = 2.9979e10 # [cm s-1]
e = 4.8023e-10 # [statcoulomb]
m_e = 9.1094e-28 # [g]
m_p = 1.6726e-24 # [g]
G = 6.6720e-8 # [dyne cm2 g-2]
k_b = 1.3807e-16 # [erg K-1]
h = 6.6261e-27 # [erg s]
R_h = 1.0974e5 # [cm-1]
a_0 = 5.2918e-9 # [cm]
r_e = 2.8179e-13 # [cm]
sigma = 5.6774e-5 # [erg cm-2 s-1 K-4]

eV = 1.6022e-12 # [erg]
SFU = 1e-19 #[erg s-1 cm-2 Hz-1]
AU = 1.5e13 # [cm]

R_sun = 6.96e10 # [cm]
M_sun =  1.99e33 # [g]
g_sun = 2.74e4 # [cm s-2]
L_sun = 3.90e33 # [erg s-1]
F_sun = 6.41e10 # [erg cm-2 s-1]
f_sun = 1.39e6 # [erg cm-2]
Omega_sun = 6.76e-5 # [ster]
T_phot = 5762 # [K]

c_p = 5./2
c_v = 3./2


mu_c = 1.27
n_c = 1e9
T_c = 1e6
B_c = 100
Z_c = 1
lnLambda_c = 20
kappa0 = 9.2e-7 # [erg cm-2 s-1 K-7/2]

gamma = c_p/c_v



def kappa(T=T_c, lnLambda=lnLambda_c):
    """Spitzer thermal conductivity [erg cm-2 s-1 K-1]"""
    #kappa0 = k_b**3.5/m_e**0.5/e**4/lnLambda # [erg cm-2 s-1 K-7/2]
    return kappa0*T**2.5

def P_th(n=n_c, T=T_c):
    """thermal pressure [dyne cm-2]"""
    return 2*k_b*n*T

def P_m(B=B_c):
    """magnetic pressure [dyne cm-2]"""
    return B**2./(8*pi)

def beta(n=n_c, T=T_c, B=B_c):
    """plasma beta parameter"""
    return P_th(n, T)/P_m(B)

def h_t(T=T_c, mu=mu_c):
    """thermal scale height [cm]"""
    return 2*k_b*T/(mu*m_p*g_sun)
    
def v_te(T=T_c):
    """electron thermal velocity [cm s-1]"""
    return np.sqrt(k_b*T/m_e)

def v_ti(T=T_c, mu=mu_c):
    """ion thermal velocity [cm s-1]"""
    return np.sqrt(k_b*T/(mu*m_p))

def ro(n=n_c, mu=mu_c):
    """ion mass density [g cm-3]"""
    return mu*n*m_p
    
def c_s(T=T_c, mu=mu_c):
    """sound speed [cm s-1]"""
    return np.sqrt(2*gamma*k_b*T/(mu*m_p))

def v_a(n=n_c, B=B_c, mu=mu_c):
    """Alfven speed [cm s-1]"""
    return B/np.sqrt(4*pi*mu*m_p*n)
    
def f_pe(n=n_c):
    """electron plasma frequency [Hz]"""
    return np.sqrt(n*e**2/(pi*m_e))
    
def f_pi(n=n_c, Z=Z_c, mu=mu_c):
    """ion plasma frequency [Hz]"""
    return np.sqrt(n*Z**2*e**2/(pi*mu*m_p))

def f_ge(B=B_c):
    """electron gyrofrequency [Hz]"""
    return e*B/(2*pi*m_e*c)
    
def f_gi(B=B_c, Z=Z_c, mu=mu_c):
    """ion gyrofrequency [Hz]"""
    return Z*e*B/(2*pi*mu*m_p*c)
    
def f_ce(n=n_c, T=T_c, lnLambda=lnLambda_c):
    """electron collision frequency [Hz]"""
    c = 4*(2*np.pi)**0.5/3*e**4/m_e**0.5/k_b**1.5
    return c*n*lnLambda/T**1.5
    
def f_ci(n=n_c, T=T_c, Z=Z_c, mu=mu_c, lnLambda=lnLambda_c):
    """ion collision frequency [Hz]"""
    c = 4*np.pi**0.5/3*e**4/(m_p*mu)**0.5/k_b**1.5
    return c*n*lnLambda*Z**4/T**1.5
    
def tau_ce(n=n_c, T=T_c, lnLambda=lnLambda_c):
    """electron collision time [s]"""
    return 1/f_ce(n, T, lnLambda)

def tau_ci(n=n_c, T=T_c, Z=Z_c, lnLambda=lnLambda_c):
    """ion collision time [s]"""
    return 1/f_ci(n, T, Z, lnLambda)

def R_e(T=T_c, B=B_c):
    """electron gyroradius [cm]"""
    return v_te(T)/(2*pi*f_ge(B))
    
def R_i(T=T_c, B=B_c, Z=Z_c, mu=mu_c):
    """ion gyroradius [cm]"""
    return v_ti(T, mu)/(2*pi*f_gi(B, Z, mu))
    
def lambda_d(n=n_c, T=T_c):
    """Debye length [cm]"""
    return np.sqrt(k_b*T/(4*pi*n*e**2))
    
def E_d(n=n_c, T=T_c, Z=Z_c, lnLambda=lnLambda_c):
    """Dreicer field [statvolt cm-1]"""
    return Z*e*lnLambda/lambda_d(n, T)
    
def sigma_e(T=T_c, Z=Z_c, lnLambda=lnLambda_c):
    """electrical conductivity [Hz]"""
    return e**2*tau_ce(1, T, Z, lnLambda)/m_e
    
def nu_m(T=T_c, Z=Z_c, lnLambda=lnLambda_c):
    """magnetic diffusivity [cm2 s-1]"""
    return c**2/(4*pi*sigma_e(T, Z, lnLambda_c))

