import os,sys
dir = os.path.dirname(__file__)
sys.path.append(dir)
import time
import threading
import numpy as np
import Sun
from utils import *
from radloss import RadLoss



class Scale():
    def __init__(self, t=1, n=1e9, T=1e6, gamma=5./3):
        self.set(t,n,T,gamma)
        
    def set(self, t, n, T, gamma):   
        self.t = t
        self.n = n
        self.T = T
        
        self.m = Sun.mu_c*Sun.m_p
        self.u = np.sqrt(2*gamma*Sun.k_b/self.m*self.T)
        self.x = self.u*self.t
        self.a = self.u/self.t
        
        self.rho = self.n*self.m
        self.rhou = self.rho*self.x/self.t
        self.rhoe = self.rhou*self.x/self.t
        
        self.rl = self.rhoe/self.n**2/self.t 
        self.kappa = self.x**2/self.T**3.5*self.rhoe/self.t

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
    def __init__(self,solver,maxiter,skip=0,each=0):
        threading.Thread.__init__(self)
        self.solver = solver
        self.maxiter = maxiter
        self.skip = skip
        self.each = each
        self.event = threading.Event()
        
    def wait(self):
        self.event.wait()
        
    def run(self):
        self.status = 'in progress'
        print(self.status+' ...')
        print('|-10%-|-20%-|-30%-|-40%-|-50%-|-60%-|-70%-|-80%-|-90%-|-100%|')    

        self.solver.niter = 0
        systime = time.time()
        
        for self.niter in range(0,self.maxiter):
            self.solver.hydrostep()
            if (self.each != 0):
                if (self.niter >= self.skip) and ((self.niter-self.skip) % self.each == 0):
                    self.solver.out.add([self.solver.q[0]/self.solver.A,self.solver.u,self.solver.T])
                    
            if (self.niter % (self.maxiter/10) == 0):
                sys.stdout.write('|')    
            if (self.niter % (self.maxiter/50) == 0):
                sys.stdout.write('x')
        print('')        
        self.status = 'done!'
        print(self.status)
        print('Elapsed time: {:.2f}'.format(time.time()-systime)+' seconds')
        self.event.set()
        
        
class Solver():
    def __init__(self,dt,X,n,u,T,hrate=lambda s,t:0,
                 gamma=5./3,btype='continuous',rl_fname='radloss.npz'):
        
        self.sc = Scale()
        self.out = Dataout()
             
        self.T0 = 2e4/self.sc.T ###
        self.rl0 = 1.52*self.sc.rhoe*self.sc.t ###
        
        self.gamma = gamma
        self.kappa = Sun.kappa/self.sc.kappa
        self.k_u = np.sqrt(self.sc.m/Sun.m_e)/12
        
        self.btype = btype

        self.time = 0  
        self.dt = dt/self.sc.t
        
        self.set_geometry(X)
        self.set_initial_values(n,u,T)
        self.set_hrate(hrate)
        self.set_radloss(rl_fname)      

        
    
    def set_geometry(self,X,h0=1e7,A=np.array([1.]),downvec=[0,0,-1]):
        self.h0 = h0/self.sc.x ###
        
        self.X = X/self.sc.x
        self.nx = self.X[0].shape[0]
        self.A = A
        Xi = np.zeros_like(self.X)
        
        for i in range(0,3):
            Xi[i] = 0.5*(self.X[i] + np.roll(self.X[i],1))
         
        self.ds = np.sqrt(np.sum([d(self.X[i])**2 for i in range(0,3)],0))
        self.ds[0] = self.h0
        
        self.dx = np.sqrt(np.sum([(np.roll(self.X[i],1)-self.X[i])**2 for i in range(0,3)],0))
        self.dx[0] = self.h0
        
        self._dx = np.roll(self.dx,-1)
        self.ddx = self.dx*self._dx*(self.dx+self._dx)
        self.dxi = np.sqrt(np.sum([(np.roll(Xi[i],-1)-Xi[i])**2 for i in range(0,3)],0))           

        self.s = np.cumsum(self.ds)
        self.L = self.s[-1]-self.s[0]
        
        self.g = np.sum([downvec[i]*d(self.X[i]) for i in range(0,3)],0)/self.dxi*Sun.g_sun/self.sc.a
        return self
        
    def set_initial_values(self,n,u,T):
        self.u = u/self.sc.u
        self.T = T/self.sc.T
        
        rho = n*self.A/self.sc.n
        rhou = n*self.u*self.A/self.sc.n
        rhoe = n*(self.u**2/2 + self.T/(self.gamma-1)/self.gamma)*self.A/self.sc.n
        
        
        self.q = [rho,rhou,rhoe]
        self.q0 = [np.copy(rho),np.copy(rhou),np.copy(rhoe)]
        return self
    
    def set_hrate(self,hrate):
        self.hrate = lambda: hrate(self.s*self.sc.x,self.time*self.sc.t)*self.A/self.sc.rhoe*self.sc.t
    
    def set_radloss(self,rl_fname='radloss.npz'):
        self.RL = RadLoss(rl_fname)
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
        return (p*np.sqrt(self.q[0])+np.roll(p*np.sqrt(self.q[0]),1))/(np.sqrt(self.q[0])+np.roll(np.sqrt(self.q[0]),1))
    
    def diffuse_imp(self):
        
        dd = self.kappa*self.A*self.T**2.5*(self.gamma-1)/2/self.q[0]
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

    def radiate(self):
        self.qn[2] -= self.dt*self.q[0]**2*self.radloss()
    
    def diffuse(self):
        D = self.interface(self.kappa*self.A*self.T**2.5)
        self.F_c = - D*(self.T-np.roll(self.T,1))/self.dx
        #self.F_clim = - self.c_si*self.k_u*self.p/(self.gamma-1)
        self.qn[2] += self.dt*(self.F_c - np.roll(self.F_c,-1))/self.dxi
        
        
    def boundary(self):
        if (self.btype == 'ebtel'):
            self.q[0][[0,-1]] = self.p[[1,-2]]/self.T0*self.gamma
            self.q[1][[0,-1]] = 0
            self.q[2][[0,-1]] = self.p[[1,-2]]/(self.gamma-1)
        
        if (self.btype == 'mirror'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q[i][[1,-2]]*(-1)**i
            
        if (self.btype == 'continuous'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q[i][[1,-2]]
            
        if (self.btype == 'periodic'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q[i][[-2,1]]
            
        if (self.btype == 'constant'):
            for i in range(0,3):
                self.q[i][[0,-1]] = self.q0[i][[0,-1]]
    
    
    def get_uTp(self):
        self.u = self.q[1]/self.q[0]
        self.T = (self.q[2]/self.q[0]-self.u**2/2)*(self.gamma-1)*self.gamma
        self.p = self.q[0]*self.T/self.gamma

        
    def advect(self):
        self.ui = self.interface(self.u)
        self.c_si = self.interface(np.sqrt(self.T))
        
        if (self.btype == 'ebtel'):
            self.ui[[1,-1]] = (- self.F_c[[1,-1]]
                               - self.rl0/self.A[[0,-1]]*self.p[[0,-1]]**2*self.h0*np.array([1,-1]))/(2.5*self.p[[0,-1]])
        
        lam = [self.ui-self.c_si,self.ui,self.ui+self.c_si]
    
        R = [[np.ones(self.nx),np.ones(self.nx),np.ones(self.nx)],
             [self.ui-self.c_si,self.ui,self.ui+self.c_si],
             [self.c_si**2/(self.gamma-1)+0.5*self.ui**2-self.c_si*self.ui,
             0.5*self.ui**2,
             self.c_si**2/(self.gamma-1)+0.5*self.ui**2+self.c_si*self.ui]]
        
        detR = 2*self.c_si**3/(self.gamma-1)          
        Rinv = [[(R[i-2][j-2]*R[i-1][j-1]-R[i-2][j-1]*R[i-1][j-2])/detR for i in range(0,3)] for j in range(0,3)]
        
        for j in range(0,3):
            flux = np.sum([R[j][i]*lam[i]*
                          np.sum([Rinv[i][k]*np.where(lam[i] >= 0., np.roll(self.q[k],1),self.q[k]) for k in range(0,3)],0)          
                          for i in range(0,3)],0)
            self.qn[j] += self.dt*(flux - np.roll(flux,-1))/self.dxi


        
    
    def hydrostep(self): 
        self.get_uTp()
        self.boundary()
        
        self.qn = np.copy(self.q)
        
        #self.radiate()
        #self.diffuse()
        self.advect()
               
        #self.qn[2] += self.dt*self.hrate()
        self.qn[1] += self.dt*self.q[0]*self.g
        self.qn[2] += self.dt*self.q[1]*self.g 
        
        self.q = np.copy(self.qn)
        
        self.time += self.dt




        
    def run(self,tau,skip=0,each=0,wait=False):
        tau /= self.sc.t
        self.mainloop = Mainloop(self,np.int(tau/self.dt),skip,each)
        self.mainloop.start()
        if wait:
            self.mainloop.wait()
        return self
        

        
        
        