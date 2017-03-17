import os,sys
dir = os.path.dirname(__file__)
import threading
import numpy as np
from .config import Scale


def average(q):
    return 0.5*(q[1:]+q[:-1])


def delta(q):
    return (q[1:]-q[:-1])


def smooth(x, d):
    from scipy.ndimage.filters import gaussian_filter1d
    #from scipy.signal import medfilt
    #return np.convolve(x, np.ones(d)/d, mode='same')
    return gaussian_filter1d(x,d,mode='wrap')
    #return medfilt(x,d)


def load(filename):
    try:
        import pickle, dill
        return pickle.load(open(filename,"rb"))
    except:
        print "can't load file"

        
def save(obj, filename):
    try:
        import pickle, dill
        pickle.dump(obj,open(filename, "wb"))
    except:
        print "can't save object"

        
        
class Data(object):
    def __init__(self, scale=Scale()):
        self.x = []
        self.q = [[],[],[]]
        self.time = []
        self.set_scale(scale)
    
    
    def set_scale(self, scale):
        self.scale = scale
        return self
    
    def add(self,time,x,q):
        for i in range(0,3):
            self.q[i] += [q[i]]
        self.x += [x]
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
        return np.array(self.x)*scale
    
    def L(self, scale=0):
        if (scale == 0):
            scale = self.scale.x
        return self.x[-1]*scale
        
    
        
class Mainloop(threading.Thread):
    def __init__(self,solver,dt,tau,data,verbose):
        threading.Thread.__init__(self)
        self.solver = solver
        self.data = data
        self.dt = dt
        self.tau = tau

        self.verbose = verbose
        self.wait_event = threading.Event()
        
        
    def wait(self):
        self.wait_event.wait()
        
        
    def run(self):
        self.status = 'in progress'
        if self.verbose:
            import time
            systime = time.time()
            
        t0 = self.solver.time
        while (self.solver.time - t0) < self.tau:
            self.solver.step(self.dt)
            
            self.data.add(self.solver.time, np.copy(self.solver.x),
                          [np.copy(self.solver.q[0]),
                           np.copy(self.solver.q[1]),
                           np.copy(self.solver.q[2])])
            
            if self.verbose:
                print '\r' + self.status + '...\t%d%%' % ((self.solver.time - t0)/self.tau*100),
               
        self.status = 'done!'
        self.wait_event.set()
        
        if self.verbose:
            print '\n'+self.status
            print 'Elapsed time: {:.2f}'.format(time.time()-systime)+' seconds'
        

        
        
        
class Solver(object):
    def __init__(self,X,n,u,T,config):
        self.set_config(config)

        self.set_x(X/self.scale.x)
        self.set_initial_values(n/self.scale.n,u/self.scale.u,T/self.scale.T)
        self.set_heating()
        self.set_gravity()
        self.set_diffusion()
        self.set_radloss() 

        self.time = 0
    
    """
    def mean(self,p,type = ''):
        if (type == ''):
            type = self.config.itype
        if (type == 'Roe'):
            return ((p*np.sqrt(self.q[0]))[:-1]+(p*np.sqrt(self.q[0]))[1:])/(np.sqrt(self.q[0][:-1])+np.sqrt(self.q[0][1:]))
        if (type == 'average'):
            return average(p)
        return p
    """
    
    def average(self,p):
        return (p[:-1]*self.dx2+p[1:]*self.dx1)/self.dx
    
    
    def d_dx(self,p,order=1):
        if (order == 1):
            return ((p[2:]*self.dx[:-1]**2-p[:-2]*self.dx[1:]**2 + p[1:-1]*(self.dx[1:]**2-self.dx[:-1]**2))/
                             (self.dx[1:]+self.dx[:-1])/self.dx[1:]/self.dx[:-1])
        if (order == 2):
            return (2*(p[2:]*self.dx[:-1]+p[:-2]*self.dx[1:] - p[1:-1]*(self.dx[1:]+self.dx[:-1]))/
                             (self.dx[1:]+self.dx[:-1])/self.dx[1:]/self.dx[:-1])
    
   
    def set_config(self, config):
        self.config = config
        self.scale = config.scale
        
        
    def set_x(self,X):
        if (len(X.shape) == 1):
            self.ndim = 1
            self.nx = X.shape[0]
        
        if (len(X.shape) == 2):
            self.ndim = X.shape[0]
            self.nx = X.shape[1]
               
        self.X = X.reshape((self.ndim,self.nx))
        self.Xi = np.zeros((self.ndim,self.nx-1),dtype=np.float32)
        
        for i in range(0,self.ndim):
            self.Xi[i] = average(self.X[i])
        
        self.update_x()
        
        
        
    def update_x(self):
        self.dx = np.sqrt(np.sum([(delta(self.X[i]))**2 for i in range(0,self.ndim)],0))
        self.dxi = np.sqrt(np.sum([(delta(self.Xi[i]))**2 for i in range(0,self.ndim)],0))
        self.dx1 = np.sqrt(np.sum([(self.Xi[i]-self.X[i][0:-1])**2 for i in range(0,self.ndim)],0))
        self.dx2 = self.dx-self.dx1
        
        self.ex = [delta(self.Xi[i])/self.dxi for i in range(0,self.ndim)]
        self.exi = [delta(self.X[i])/self.dx for i in range(0,self.ndim)]

        
        self.x = np.cumsum(np.append([0],self.dx))
        self.xi = np.cumsum(np.append([0],self.dxi))
        self.L = self.x[-1]
        
        
    
    
    def set_initial_values(self,n,u,T):
        rho = n
        rhou = n*u
        rhoe = n*(u**2/2 + T/(self.scale.gamma-1))
        
        self.w = np.copy(u)
        self.wi = self.average(self.w)
        #self.w = np.ones_like(u)
        
        self.q = [rho,rhou,rhoe]
        self.q0 = [np.copy(rho),np.copy(rhou),np.copy(rhoe)]
        self.T0 = np.copy(T)
        
        self.set_boundary()
        return self
    
    
    def set_heating(self):
        if self.config.Hr:
            self.Hr = lambda: self.config.Hr(self.x*self.scale.x,self.time*self.scale.t)/self.scale.rhoe*self.scale.t
        else:
            self.Hr = False
    
    
    def set_gravity(self): 
        if self.config.g:
            self.g = lambda: (np.sum([self.config.g[i]*self.ex[i] for i in range(0,self.ndim)],0)/self.scale.a)
        else:
            self.g = False
    
    
    def set_radloss(self):
        if self.config.Lambda:
            self.Lambda = lambda: self.config.Lambda(self.T*self.scale.T)/self.scale.rl
        else:
            self.Lambda = False
    
    
    def set_diffusion(self):
        if self.config.kappa:
            self.kappa = lambda: self.config.kappa(self.T*self.scale.T)/self.scale.kappa
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
            self.q[0][[0,-1]] = self.q[2][[0,-1]]/self.T0[[0,-1]]/(self.scale.gamma-1)
            self.q[1][[0,-1]] = 0        
            
    
    def set_cells(self):
        self.dq = [delta(self.q[i]) for i in range(0,3)]
        
        self.u = self.q[1]/self.q[0]
        self.ui = self.average(self.u)
        
        self.T = (self.q[2]/self.q[0]-self.u**2/2)*(self.scale.gamma-1)
        self.p = self.q[0]*self.T
        self.c_s = np.sqrt(self.scale.gamma*self.T)
        self.c_si = self.average(self.c_s)
        
        #self.w =  smooth(self.u,3)
        self.w =  self.u*1
        self.w[[0,-1]] = 0
        self.wi = self.average(self.w)  
            
        self.F = [self.q[0]*(self.u-self.w),self.q[1]*(self.u-self.w)+self.p,self.q[2]*(self.u-self.w)+self.p*self.u] 
        #self.F = [self.q[0]*(self.u-self.w),self.q[1]*(self.u-self.w)+self.p,(self.q[2]+self.p)*(self.u-self.w)] 
        #self.F = [self.q[0]*(self.u-self.w),self.q[1]*(self.u-self.w),self.q[2]*(self.u-self.w)]
                         
        if self.kappa:
            self.D = self.kappa()
            
    
    
    def get_substep(self, dt):
        self.cfl = [np.max((np.abs((self.ui-self.wi))/self.dx)),
                    np.max((np.abs(self.c_si)/self.dx))]
        
        if self.kappa:
            self.cfl = self.cfl + [np.max((np.abs(self.Di)/self.dx**2))]
        
        return np.ceil(np.max(self.cfl)*dt/self.config.cfl_lim).astype(int)
                
    def set_fluxes(self):
        self.Fi = [self.average(self.F[i]) for i in range(0,3)]
        
        if self.kappa:
            self.Di = self.average(self.D)
            self.Fci = self.Di*delta(self.T)/self.dx
            self.Fi[2] -= self.Fci
    
    
    def set_correction(self):
        ui = self.ui
        wi = self.wi
        c_si = self.c_si
        
        if (self.config.ctype == 'upwind'):
            for i in range(0,3):
                self.Fi[i] -= 0.5*np.abs(ui-wi)*self.dq[i]
                
        if (self.config.ctype == 'Riemann'):
            gamma = self.scale.gamma
            
            one = np.ones(self.nx-1)
            lam = [ui-c_si-wi,ui-wi,ui+c_si-wi]
            #lam = [ui-c_si,ui,ui+c_si]
            detR = 2*c_si**2/(gamma-1)
            R = [[one,one,one],
                 [ui-c_si,ui,ui+c_si],
                 [0.5*detR+0.5*ui**2-c_si*ui,
                  0.5*ui**2,
                  0.5*detR+0.5*ui**2+c_si*ui]]

            R1 = [[c_si*ui/(gamma-1)+0.5*ui**2,-c_si/(gamma-1)-ui,one],
                  [detR-ui**2,2*ui,-2*one],
                  [-c_si*ui/(gamma-1)+0.5*ui**2,c_si/(gamma-1)-ui,one]]
            
            ai = [np.sum([R1[i][j]/detR*self.dq[j] for j in range(0,3)],0) for i in range(0,3)]
            Wi = [[R[i][j]*ai[j] for j in range(0,3)] for i in range(0,3)]

            for i in range(0,3):       
                Ai = 0.5*np.sum([np.abs(lam[j])*Wi[i][j] for j in range(0,3)],0) 
                self.Fi[i] -= Ai      
          
                
    
    def advect(self,dt):
        for i in range(0,3):
            self.q[i][1:-1] -= dt*delta(self.Fi[i])/self.dxi
            
    def step(self,dt):
        self.set_cells()
        self.set_fluxes()
        self.set_correction()
        n_s = self.get_substep(dt)
        
        self.advect(dt/n_s)

        if self.Lambda:
            self.q[2][1:-1] -= dt/n_s*self.q[0][1:-1]**2*self.Lambda()[1:-1]
        if self.Hr:
            self.q[2][1:-1] += dt/n_s*self.Hr()[1:-1]
        if self.g:
            self.q[2][1:-1] += dt/n_s*self.q[1][1:-1]*self.g()
            self.q[1][1:-1] += dt/n_s*self.q[0][1:-1]*self.g()   
        
        for i in range(0,self.ndim):
            self.X[i][1:-1] += dt/n_s*self.w[1:-1]*self.ex[i]
            self.Xi[i] += dt/n_s*self.wi*self.exi[i]
            self.update_x()
        
        for i in range(0,3):
            self.q[i][1:-1] -= dt/n_s*self.q[i][1:-1]*self.d_dx(self.w)
 
        #self.w[1:-1] += dt/n_s*self.d_dx(self.p)/self.q[0][1:-1]
        #self.wi += dt/n_s*delta(self.p)/self.dx/self.average(self.q[0])
        
        self.time += dt/n_s
        self.set_boundary()
        
        if (n_s > 1):
            for i in range(0,n_s-1):
                self.step(dt/n_s)


    def run(self,dt,tau,data=Data(),wait=True,verbose=True):
        data.set_scale(self.scale)
        mainloop = Mainloop(self,dt/self.scale.t,tau/self.scale.t,data,verbose)
        mainloop.start()
        if wait:
            mainloop.wait()
        return data
    
    

        
        

        
        
        