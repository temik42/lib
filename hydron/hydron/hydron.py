import os,sys
dir = os.path.dirname(__file__)
sys.path.append(dir)
import time
import threading
import numpy as np
from . import Sun
from .config import defconfig


class Scale():
    def __init__(self, config):
        self.m = config.m
        self.k_b = config.k_b
        self.gamma = config.gamma
        
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
        
        self.rl = self.rhoe/self.n**2/self.t
        self.kappa = self.x**2/self.T*self.rhoe/self.t

class Data():
    def __init__(self, x, scale):
        self.scale = scale
        self.x = x
        self.q = [[],[],[]]
        self.time = []
        
    def add(self,time,q):
        for i in range(0,3):
            self.q[i] += [q[i]]
        self.time += [time]
            
    def n(self, scale=0):
        if (scale == 0):
            scale = self.scale.n
        return np.array(self.q[0])*scale

    def u(self, scale=0):
        if (scale == 0):
            scale = self.scale.u
        return np.array(self.q[1])/self.n(1)*scale
    
    def T(self, scale=0):
        if (scale == 0):
            scale = self.scale.T
        return (self.q[2]/self.n(1)-self.u(1)**2/2)*(self.scale.gamma-1)*scale
    
    def tau(self, scale=0):
        if (scale == 0):
            scale = self.scale.t
        return self.time[-1]*scale
    
    def s(self, scale=0):
        if (scale == 0):
            scale = self.scale.x
        return self.x*scale
    
    def L(self, scale=0):
        if (scale == 0):
            scale = self.scale.x
        return self.x[-1]*scale
        
    
        
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
                self.solver.out.add(self.solver.time,
                                    [self.solver.q[0]/self.solver.A,self.solver.q[1]/self.solver.A,self.solver.q[2]/self.solver.A])
            
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
    def __init__(self,dt,X,n,u,T, config = defconfig()):
        self.config = config
        self.sc = Scale(self.config)
        
        self.time = 0  
        self.dt = dt/self.sc.t
        
        self.set_geometry(X)
        self.set_initial_values(n,u,T)
        self.set_heating()
        self.set_gravity()
        self.set_diffusion()
        self.set_radloss() 
        
        self.out = Data(self.s,self.sc)
    
    
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
    
    
    def mean(self,p,type = ''):
        if (type == ''):
            type = self.config.itype
        if (type == 'Roe'):
            return (p*np.sqrt(self.q[0])+np.roll(p*np.sqrt(self.q[0]),1))/(np.sqrt(self.q[0])+np.roll(np.sqrt(self.q[0]),1))
        if (type == 'average'):
            return (p+np.roll(p,1))*0.5
        return p
    
    
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
        self.Xi = np.zeros_like(self.X)
        
        for i in range(0,self.ndim):
            self.Xi[i] = 0.5*(self.X[i] + np.roll(self.X[i],1))
         
        self.ds = np.sqrt(np.sum([d(self.X[i])**2 for i in range(0,self.ndim)],0))
        self.ds[0] = 0
        
        self.dx = np.sqrt(np.sum([(np.roll(self.X[i],1)-self.X[i])**2 for i in range(0,self.ndim)],0))
        self.dx[0] = self.dx[1]
        
        self._dx = np.roll(self.dx,-1)
        self.ddx = self.dx*self._dx*(self.dx+self._dx)
        self.dxi = np.sqrt(np.sum([(np.roll(self.Xi[i],-1)-self.Xi[i])**2 for i in range(0,self.ndim)],0))           

        self.s = np.cumsum(self.ds)
        self.L = self.s[-1]-self.s[0]
       
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
    
    
    def set_heating(self):
        if self.config.Hr:
            self.Hr = lambda: self.config.Hr(self.s*self.sc.x,self.time*self.sc.t)*self.A/self.sc.rhoe*self.sc.t
        else:
            self.Hr = False
    
    
    def set_gravity(self): 
        if self.config.g:
            self.g = lambda: (np.sum([self.config.g[i]*(np.roll(self.Xi[i],-1)-self.Xi[i]) for i in range(0,self.ndim)],0)/
                              self.dxi/self.sc.a) #
        else:
            self.g = False
    
    
    def set_radloss(self):
        if self.config.Lambda:
            self.Lambda = lambda: self.config.Lambda(self.T*self.sc.T)/self.sc.rl/self.A
        else:
            self.Lambda = False
    
    
    def set_diffusion(self):
        if self.config.kappa:
            self.kappa = lambda: self.config.kappa(self.T*self.sc.T)/self.sc.kappa*self.A
        else:
            self.kappa = False

        
    def set_boundary(self):
      
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
    
        if (self.config.btype == 'stationary'):
            self.q[2][[0,-1]] = self.q[2][[1,-2]]-0.5*self.q[1][[1,-2]]**2/self.q[0][[1,-2]]
            self.q[0][[0,-1]] = self.q[2][[0,-1]]/self.config.T0/(self.config.gamma-1)
            self.q[1][[0,-1]] = 0        
            
    
    def set_cells(self):
        self.dq = [self.q[i]-np.roll(self.q[i],1) for i in range(0,3)]
        self.u = self.q[1]/self.q[0]
        self.T = (self.q[2]/self.q[0]-self.u**2/2)*(self.config.gamma-1)
        self.p = self.q[0]*self.T
        self.F = [self.q[1],self.q[1]*self.u+self.p,(self.q[2]+self.p)*self.u]       

        
    def set_fluxes(self):    
        self.Fi = [self.mean(self.F[i],'average') for i in range(0,3)]
        
        if self.kappa:
            Di = self.mean(self.kappa())
            Fc = Di*(self.T-np.roll(self.T,1))/self.dx
            self.Fi[2] -= Fc
    
    
    def set_correction(self):
        if (self.config.ctype == 'upwind'):
            ui = self.mean(self.u)
            for i in range(0,3):
                self.Fi[i] -= 0.5*np.abs(ui)*self.dq[i]
                
        if (self.config.ctype == 'Riemann'):
            gamma = self.config.gamma
            ui = self.mean(self.u)
            c_si = self.mean(np.sqrt(gamma*self.T))
            one = np.ones(self.nx)
            lam = [ui-c_si,ui,ui+c_si]
            detR = 2*c_si**2/(gamma-1)
            R = [[one,one,one],
                 lam,
                 [0.5*detR+0.5*ui**2-c_si*ui,
                 0.5*ui**2,
                 0.5*detR+0.5*ui**2+c_si*ui]]

            R1 = [[c_si*ui/(gamma-1)+0.5*ui**2,-c_si/(gamma-1)-ui,one],
                    [detR-ui**2,2*ui,-2*one],
                    [-c_si*ui/(gamma-1)+0.5*ui**2,c_si/(gamma-1)-ui,one]]

            for i in range(0,3):       
                Ai = 0.5*np.sum([R[i][j]*np.abs(lam[j]/detR)*np.sum([R1[j][k]*self.dq[k] for k in range(0,3)],0)          
                              for j in range(0,3)],0)             
                self.Fi[i] -= Ai
 
    
    def advect(self):
        for i in range(0,3):
            self.q[i] -= self.dt*(np.roll(self.Fi[i],-1)-self.Fi[i])/self.dxi

            
    def step(self):
        self.set_boundary()
        self.set_cells()
        self.set_fluxes()
        self.set_correction()
        
        self.advect()

        if self.Lambda:
            self.q[2] -= self.dt*self.q[0]**2*self.Lambda()
        if self.Hr:
            self.q[2] += self.dt*self.Hr()
        if self.g:
            self.q[2] += self.dt*self.q[1]*self.g() 
            self.q[1] += self.dt*self.q[0]*self.g()   
   
        self.time += self.dt

        
    def run(self,tau,each=0,skip=0):
        tau /= self.sc.t
        if (each == 0):
            each = tau/(100*self.dt)
        self.mainloop = Mainloop(self,int(tau/self.dt),int(skip),int(each),self.config.verbose)
        self.mainloop.start()
        if self.config.wait:
            self.mainloop.wait()
        return self
        

        
        
        