import os,sys
dir = os.path.dirname(__file__)
sys.path.append(dir)
import time
import threading
import numpy as np
#from numpy import dot
#from numpy.linalg import inv, eig
from . import Sun
#from .utils import *
from .config import defconfig
from .radloss import RadLoss



class Scale():
    def __init__(self, config):
        self.m = config.m
        self.k_b = config.k_b
        
        if (config.sctype == 'noscale'):
            self.set(1,1,1)
            
        if (config.sctype == 'coronal'):
            self.set(1,1e9,1e6)
        
        if (config.sctype == 'chromospheric'):
            self.set(1,1e11,1e4)
        
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
        
        self.rl = self.rhoe/self.n**2/self.t #
        self.kappa = self.x**2/self.T**3.5*self.rhoe/self.t #

class Dataout():
    def __init__(self):
        self.n = []
        self.u = []
        self.T = []
        
    def add(self,q):
        self.n += [q[0]]
        self.u += [q[1]]
        self.T += [q[2]]
        
class Mainloop(threading.Thread):
    def __init__(self,solver,maxiter,skip=0,each=1,verbose=True):
        threading.Thread.__init__(self)
        self.solver = solver
        self.maxiter = maxiter
        self.skip = skip
        self.each = each
        self.verbose = verbose
        self.event = threading.Event()
        
        
    def wait(self):
        self.event.wait()
        
    def run(self):
        self.status = 'in progress'
        if self.verbose:
            print(self.status+' ...')
            print('|-10%-|-20%-|-30%-|-40%-|-50%-|-60%-|-70%-|-80%-|-90%-|-100%|')    

        systime = time.time()
        
        for self.niter in range(0,self.maxiter):
            self.solver.step()
            
            if (self.niter >= self.skip) and (int(self.niter-self.skip) % int(self.each) == 0):
                self.solver.out.add([self.solver.q[0]/self.solver.A,self.solver.u,self.solver.T])
            
            if self.verbose:
                if (self.niter % (self.maxiter/10) == 0):
                    sys.stdout.write('|')    
                if (self.niter % (self.maxiter/50) == 0):
                    sys.stdout.write('x')
                
        self.status = 'done!'
        self.event.set()
        
        if self.verbose:
            print('')
            print(self.status)
            print('Elapsed time: {:.2f}'.format(time.time()-systime)+' seconds')
        
        
        
class Solver():
    def __init__(self,dt,X,n,u,T,hrate=lambda s,t:0, config = defconfig()):
        self.config = config
        
        self.sc = Scale(self.config)
        self.out = Dataout()
        
        if (self.config.btype == 'ebtel'):
            self.T0 = self.config.T0/self.sc.T #
            self.rl0 = self.config.rl0*self.sc.rhoe*self.sc.t #
            self.h0 = self.config.h0/self.sc.x #
            self.set_radloss() #
        else:
            self.h0 = 0
        
            
        self.kappa = Sun.kappa/self.sc.kappa #
        
        self.time = 0  
        self.dt = dt/self.sc.t
        
        self.set_geometry(X)
        self.set_initial_values(n,u,T)
        self.set_hrate(hrate)
   

        
    
    def set_geometry(self,X,A=np.array([1.])):
        d = lambda x: np.gradient(x,edge_order = 1)        
        
        if (len(X.shape) == 1):
            self.ndim = 1
            self.nx = X.shape[0]
        
        if (len(X.shape) == 2):
            self.ndim = X.shape[0]
            self.nx = X.shape[1]
               
        self.X = X.reshape((self.ndim,self.nx))/self.sc.x       
        self.A = A
        Xi = np.zeros_like(self.X)
        
        for i in range(0,self.ndim):
            Xi[i] = 0.5*(self.X[i] + np.roll(self.X[i],1))
         
        self.ds = np.sqrt(np.sum([d(self.X[i])**2 for i in range(0,self.ndim)],0))
        self.ds[0] = 0
        
        self.dx = np.sqrt(np.sum([(np.roll(self.X[i],1)-self.X[i])**2 for i in range(0,self.ndim)],0))
        self.dx[0] = self.h0
        
        self._dx = np.roll(self.dx,-1)
        self.ddx = self.dx*self._dx*(self.dx+self._dx)
        self.dxi = np.sqrt(np.sum([(np.roll(Xi[i],-1)-Xi[i])**2 for i in range(0,self.ndim)],0))           

        self.s = np.cumsum(self.ds)
        self.L = self.s[-1]-self.s[0]
        
        self.g = np.sum([self.config.downvec[i]*d(self.X[i]) for i in range(0,self.ndim)],0)/self.dxi*self.config.g/self.sc.a #
        return self
        
    def set_initial_values(self,n,u,T):
        self.u = u/self.sc.u
        self.T = T/self.sc.T
        
        rho = n*self.A/self.sc.n
        rhou = n*self.u*self.A/self.sc.n
        rhoe = n*(self.u**2/2 + self.T/(self.config.gamma-1))*self.A/self.sc.n
        
        
        self.q = [rho,rhou,rhoe]
        self.q0 = [np.copy(rho),np.copy(rhou),np.copy(rhoe)]
        return self
    
    def set_hrate(self,hrate):
        self.hrate = lambda: hrate(self.s*self.sc.x,self.time*self.sc.t)*self.A/self.sc.rhoe*self.sc.t
    
    def set_radloss(self):
        self.RL = RadLoss(self.config.rl_fname)
        self.radloss = lambda: self.RL.get(self.T*self.sc.T)/self.sc.rl/self.A
        
    def d_dx(self,q,order = 1):
        if (order == 1):
            out = (self.dx**2*np.roll(q,-1)+(self._dx**2-self.dx**2)*q-self._dx**2*np.roll(q,1))/self.ddx
            out[0] = (-(self.dx[2]**2+2*self.dx[1]*self.dx[2])*q[0]+(self.dx[1]+self.dx[2])**2*q[1]-self.dx[1]**2*q[2])/self.ddx[1]
            out[-1] = (self.dx[-1]**2*q[-3]-(self.dx[-2]+self.dx[-1])**2*q[-2] + 
                       (self.dx[-2]**2+2*self.dx[-1]*self.dx[-2])*q[-1])/self.ddx[-2]
            return out
        if (order == 2):
            out = 2*(self._dx*np.roll(q,1)-(self.dx+self._dx)*q+self.dx*np.roll(q,-1))/self.ddx
            out[[0,-1]] = out[[1,-2]]
            return out
      
    def interface(self,p):
        if (self.config.itype == 'Roe'):
            return (p*np.sqrt(self.q[0])+np.roll(p*np.sqrt(self.q[0]),1))/(np.sqrt(self.q[0])+np.roll(np.sqrt(self.q[0]),1))
        if (self.config.itype == 'average'):
            return (p+np.roll(p,1))*0.5
        return p
    """    
    def diffuse_imp(self):
        
        dd = self.kappa*self.A*self.T**2.5*(self.config.gamma-1)/2/self.q[0]
        z = 5./7*self.kappa*self.A*self.T**3.5
        q = self.q[2]-self.q[1]**2/(2*self.q[0])
        
        D = -1/self.dt-2*dd/(self.dx*self._dx)
        D[[0,-1]] = 1
        
        U = 2*np.roll(dd,-1)/(self._dx*(self.dx+self._dx))
        U = U[:-1]
        U[0] = 0

        V = 2*np.roll(dd,1)/(self.dx*(self.dx+self._dx))
        V = V[1:]
        V[-1] = 0

        B = self.d_dx(z,2) - q/self.dt
        B[[0,-1]] = q[[0,-1]]
        
        self.q[2] += trisol(D,U,V,B) - q
    """
    def radiate(self):
        self.qn[2] -= self.dt*self.q[0]**2*self.radloss()
    
    def diffuse(self):
        D = self.interface(self.kappa*self.A*self.T**2.5)
        self.F_c = - D*(self.T-np.roll(self.T,1))/self.dx
        self.qn[2] += self.dt*(self.F_c - np.roll(self.F_c,-1))/self.dxi
        
        
    def boundary(self):
        if (self.config.btype == 'ebtel'):
            self.q[0][[0,-1]] = self.p[[1,-2]]/self.T0
            self.q[1][[0,-1]] = 0
            self.q[2][[0,-1]] = self.p[[1,-2]]/(self.config.gamma-1)
        
        if (self.config.btype == 'mirror'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q[i][[1,-2]]*(-1)**i
            
        if (self.config.btype == 'continuous'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q[i][[1,-2]]
            
        if (self.config.btype == 'periodic'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q[i][[-2,1]]
            
        if (self.config.btype == 'constant'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q0[i][[0,-1]]
    
    
    def get_uTp(self):
        self.u = self.q[1]/self.q[0]
        self.e = self.q[2]/self.q[0]
        self.T = (self.e-self.u**2/2)*(self.config.gamma-1)
        self.p = self.q[0]*self.T

        
    def advect(self):
        ui = self.interface(self.u)
        c_si = self.interface(np.sqrt(self.config.gamma*self.T))
        #ei = self.interface(self.e)
        one = np.ones(self.nx)
        #zero = np.zeros(self.nx)
        gamma = self.config.gamma
        
        if (self.config.btype == 'ebtel'):
            ui[[1,-1]] = (- self.F_c[[1,-1]]
                               - self.rl0/self.A[[0,-1]]*self.p[[0,-1]]**2*self.h0*np.array([1,-1]))/(2.5*self.p[[0,-1]])
        
        lam = [ui-c_si,ui,ui+c_si]
        
        R = [[one,one,one],
             [ui-c_si,ui,ui+c_si],
             [c_si**2/(gamma-1)+0.5*ui**2-c_si*ui,
             0.5*ui**2,
             c_si**2/(gamma-1)+0.5*ui**2+c_si*ui]]
        
        detR = 2*c_si**3/(gamma-1)          
        Rinv = [[(R[i-2][j-2]*R[i-1][j-1]-R[i-2][j-1]*R[i-1][j-2])/detR for i in range(0,3)] for j in range(0,3)]
              
        
        for j in range(0,3):
            
            Wi = [R[j][i]*lam[i]*
                          np.sum([Rinv[i][k]*(self.q[k]-np.roll(self.q[k],1)) for k in range(0,3)],0)          
                          for i in range(0,3)]
            
            self.qn[j] -= self.dt*np.sum([np.where(lam[i] >= 0, Wi[i],np.roll(Wi[i],-1)) for i in range(0,3)],0)/self.dxi
            
   
    
    def step(self): 
        self.get_uTp()
        self.boundary()
        
        self.qn = np.copy(self.q)
        
        #self.radiate()
        #self.diffuse()
        self.advect()
               
        #self.qn[2] += self.dt*self.hrate()
        #self.qn[1] += self.dt*self.q[0]*self.g
        #self.qn[2] += self.dt*self.q[1]*self.g 
        
        self.q = np.copy(self.qn)
        
        self.time += self.dt




        
    def run(self,tau,each=0,skip=0,wait=True,verbose=True):
        tau /= self.sc.t
        if (each == 0):
            each = tau/(100*self.dt)
        self.mainloop = Mainloop(self,int(tau/self.dt),int(skip),int(each),verbose)
        self.mainloop.start()
        if wait:
            self.mainloop.wait()
        return self
        

        
        
        